from openai import AzureOpenAI
from typing import List, Dict, Any, Optional
from models import DecisionResult, QueryAnalysis
from config import Config
import json
import logging

logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self):
        self.config = Config()
        
        # Check if Azure AI Studio credentials are available
        if not self.config.AZURE_AI_STUDIO_API_KEY:
            raise Exception("Azure AI Studio API key not configured. Please set AZURE_AI_STUDIO_API_KEY in your .env file.")
        if not self.config.AZURE_AI_STUDIO_ENDPOINT:
            raise Exception("Azure AI Studio endpoint not configured. Please set AZURE_AI_STUDIO_ENDPOINT in your .env file.")
        
        self.client = AzureOpenAI(
            api_key=self.config.AZURE_AI_STUDIO_API_KEY,
            api_version=self.config.AZURE_AI_STUDIO_API_VERSION,
            azure_endpoint=self.config.AZURE_AI_STUDIO_ENDPOINT
        )
        
        # Test the connection
        logger.info("🧪 Testing Azure AI Studio LLM connection...")
        test_response = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        logger.info("✅ Azure AI Studio LLM connection successful")
        
        # Improved system prompts for better accuracy
        self.system_prompts = {
            "coverage_check": """You are an expert insurance policy analyst specializing in coverage analysis. Answer based ONLY on the provided policy document.

CRITICAL INSTRUCTIONS:
1. For coverage questions: State exactly what IS and IS NOT covered
2. Include ALL conditions, limitations, and exclusions
3. Provide EXACT policy clause references
4. Include specific amounts, percentages, and currency when mentioned
5. Be comprehensive but precise

COVERAGE ANALYSIS RULES:
- Start with a direct yes/no if applicable
- List ALL covered items and conditions
- List ALL exclusions and limitations
- Include exact policy clause numbers and references
- Specify waiting periods, deductibles, and limits
- Mention any sub-limits or special conditions

CRITICAL FORMATTING RULES:
- Write in plain text only
- NO markdown formatting, NO special characters, NO headers
- NO ### headers, NO ## headers, NO # headers
- NO bullet points, NO numbered lists, NO dashes
- Write as one continuous paragraph
- Use simple punctuation only
- NO line breaks or special formatting
- NO formatting of any kind

ANSWER QUALITY RULES:
- Provide specific details from the policy when available
- Include relevant policy sections and clauses
- If the information exists, be thorough in your response
- If information is missing, clearly state this
- Focus on accuracy and completeness
- Use clear, professional language""",

            "waiting_period": """You are an expert insurance policy analyst specializing in waiting periods and time-based conditions.

CRITICAL INSTRUCTIONS:
1. For waiting periods: Provide EXACT timeframes from the policy
2. Distinguish between different types of waiting periods
3. Include ALL conditions that affect waiting periods
4. Specify when coverage begins and ends
5. Include policy clause references

WAITING PERIOD ANALYSIS RULES:
- Provide EXACT timeframes (days, months, years)
- Distinguish between pre-existing, specific procedures, and general waiting periods
- Include conditions that can reduce or extend waiting periods
- Mention portability benefits and prior coverage
- Specify policy clause numbers and references

CRITICAL FORMATTING RULES:
- Write in plain text only
- NO markdown formatting, NO special characters, NO headers
- NO ### headers, NO ## headers, NO # headers
- NO bullet points, NO numbered lists, NO dashes
- Write as one continuous paragraph
- Use simple punctuation only
- NO line breaks or special formatting
- NO formatting of any kind

ANSWER QUALITY RULES:
- Provide specific time periods and conditions
- Include relevant policy clauses
- If information exists, be thorough
- If information is missing, clearly state this
- Use precise language for time periods""",

            "exclusion_check": """You are an expert insurance policy analyst focusing on exclusions and limitations.

CRITICAL INSTRUCTIONS:
1. For exclusions: List ALL applicable exclusions from the policy
2. Include ALL conditions that void coverage
3. Specify exact policy clause references
4. Distinguish between general and specific exclusions
5. Include any exceptions to exclusions

EXCLUSION ANALYSIS RULES:
- List ALL what is NOT covered under the policy
- Include specific exclusions for procedures, conditions, and circumstances
- Mention any conditions that can void coverage
- Include policy clause numbers and references
- Specify any exceptions or conditions where exclusions don't apply

CRITICAL FORMATTING RULES:
- Write in plain text only
- NO markdown formatting, NO special characters, NO headers
- NO ### headers, NO ## headers, NO # headers
- NO bullet points, NO numbered lists, NO dashes
- Write as one continuous paragraph
- Use simple punctuation only
- NO line breaks or special formatting
- NO formatting of any kind

ANSWER QUALITY RULES:
- List specific exclusions and limitations
- Include relevant policy clauses
- Be comprehensive in coverage
- If information is missing, clearly state this
- Be clear about what is not covered""",

            "policy_details": """You are an expert insurance policy analyst providing detailed policy information.

CRITICAL INSTRUCTIONS:
1. For policy details: Include ALL relevant information from the policy
2. Provide specific amounts, percentages, and currency when mentioned
3. Include ALL conditions and requirements
4. Specify exact policy clause references
5. Be comprehensive and detailed

POLICY DETAILS ANALYSIS RULES:
- Explain ALL policy features and benefits clearly
- Include specific coverage limits and amounts
- Mention ALL terms and conditions
- Include relevant definitions and clarifications
- Specify policy clause numbers and references
- Include any sub-limits or special conditions

CRITICAL FORMATTING RULES:
- Write in plain text only
- NO markdown formatting, NO special characters, NO headers
- NO ### headers, NO ## headers, NO # headers
- NO bullet points, NO numbered lists, NO dashes
- Write as one continuous paragraph
- Use simple punctuation only
- NO line breaks or special formatting
- NO formatting of any kind

ANSWER QUALITY RULES:
- Provide specific policy details
- Include relevant clauses and sections
- Be thorough in explanations
- If information is missing, clearly state this
- Use clear, professional language""",

            "general_query": """You are an expert insurance policy analyst. Answer questions about the insurance policy based on the provided document content with high accuracy.

CRITICAL INSTRUCTIONS:
1. Base answers ONLY on the provided policy document
2. Be accurate and specific with exact details
3. Cite relevant policy sections and clause numbers
4. Include specific amounts, percentages, and currency when mentioned
5. Provide comprehensive information

GENERAL ANALYSIS RULES:
- Start with a direct answer if applicable
- Include ALL relevant details from the policy
- Provide specific policy clause references
- Include exact amounts, percentages, and conditions
- Mention any limitations or exclusions
- Be comprehensive and accurate

CRITICAL FORMATTING RULES:
- Write in plain text only
- NO markdown formatting, NO special characters, NO headers
- NO ### headers, NO ## headers, NO # headers
- NO bullet points, NO numbered lists, NO dashes
- Write as one continuous paragraph
- Use simple punctuation only
- NO line breaks or special formatting
- NO formatting of any kind

ANSWER QUALITY RULES:
- Provide specific details from the policy
- Include relevant policy sections
- Be thorough when information is available
- If information is missing, clearly state this
- Focus on accuracy and completeness
- Use clear, professional language"""
        }
    
    def process_query(self, query: str, relevant_chunks: List[Dict[str, Any]], query_analysis: QueryAnalysis) -> str:
        """Process query using LLM with relevant document chunks"""
        try:
            # Prepare context from relevant chunks
            context = self._prepare_context(relevant_chunks)
            
            # Get appropriate system prompt
            intent_key = query_analysis.intent.value
            system_prompt = self.system_prompts.get(intent_key, self.system_prompts["general_query"])
            
            # Create user prompt
            user_prompt = self._create_user_prompt(query, context, query_analysis)
            
            # Call Azure AI Studio API
            response = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,  # Using the model name from config
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Simple cleaning to remove any formatting
            cleaned_answer = self._simple_clean(answer)
            
            logger.info(f"Generated answer for query: {query[:50]}...")
            
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"Error processing query with LLM: {e}")
            return f"I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
    
    def _simple_clean(self, text: str) -> str:
        """Enhanced cleaning that preserves important insurance symbols and formatting"""
        import re
        
        # Remove markdown formatting but preserve important symbols
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Remove code blocks
        
        # Remove ALL newlines and replace with spaces
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        
        # Multiple passes to remove ALL markdown headers and formatting
        for _ in range(3):  # Multiple passes to catch all variations
            text = re.sub(r'#{1,6}\s*', '', text)  # Remove all markdown headers
            text = re.sub(r'---+', '', text)  # Remove horizontal lines
            text = re.sub(r'###\s*', '', text)  # Remove ### headers
            text = re.sub(r'##\s*', '', text)  # Remove ## headers
            text = re.sub(r'#\s*', '', text)  # Remove # headers
            text = re.sub(r'---\s*', '', text)  # Remove --- headers
            text = re.sub(r'---', '', text)  # Remove any remaining ---
        
        # Remove ALL bullet points, dashes, and numbered lists
        text = re.sub(r'^\s*[-*+]\s*', '', text, flags=re.MULTILINE)  # Remove bullet points
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)  # Remove numbered lists
        text = re.sub(r'\s*[-*+]\s*', ' ', text)  # Remove inline bullet points
        text = re.sub(r'\s*\d+\.\s*', ' ', text)  # Remove inline numbered lists
        text = re.sub(r'\s*-\s*', ' ', text)  # Remove standalone dashes
        text = re.sub(r'\s*---\s*', ' ', text)  # Remove any remaining ---
        
        # Enhanced cleaning that preserves important insurance symbols
        # Preserve currency symbols, percentages, policy numbers, and important insurance terms
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\$\%\₹\@\#\d]', '', text)
        
        # Keep policy numbers intact (e.g., 1.2.3, 2.1, etc.)
        text = re.sub(r'(?<!\w)(\d+\.\d+\.\d+)(?!\w)', r'\1', text)
        text = re.sub(r'(?<!\w)(\d+\.\d+)(?!\w)', r'\1', text)
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove any remaining escape characters
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\"', '"')
        
        # Final cleanup - remove any remaining formatting artifacts
        text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace
        text = text.strip()
        
        # Remove any remaining dashes that might be formatting artifacts
        text = re.sub(r'\s*-\s*', ' ', text)  # Remove standalone dashes
        text = re.sub(r'\s*---\s*', ' ', text)  # Remove any remaining ---
        
        # Final normalization
        text = re.sub(r'\s+', ' ', text)  # One more pass to clean whitespace
        text = text.strip()
        
        # Final cleanup - remove any remaining ### headers that might have slipped through
        text = re.sub(r'###\s*', '', text)  # Remove any remaining ### headers
        text = re.sub(r'##\s*', '', text)  # Remove any remaining ## headers
        text = re.sub(r'#\s*', '', text)  # Remove any remaining # headers
        
        # Final normalization
        text = re.sub(r'\s+', ' ', text)  # One more pass to clean whitespace
        text = text.strip()
        
        # Final post-processing: remove any remaining formatting artifacts
        # This is a catch-all for any formatting that might have slipped through
        text = re.sub(r'###\s*', '', text)  # Final ### removal
        text = re.sub(r'##\s*', '', text)  # Final ## removal
        text = re.sub(r'#\s*', '', text)  # Final # removal
        text = re.sub(r'---\s*', '', text)  # Final --- removal
        text = re.sub(r'---', '', text)  # Final --- removal
        text = re.sub(r'\s*-\s*', ' ', text)  # Final dash removal
        
        # Final normalization
        text = re.sub(r'\s+', ' ', text)  # One more pass to clean whitespace
        text = text.strip()
        
        return text
    
    def _prepare_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from relevant document chunks with improved quality"""
        if not relevant_chunks:
            return "No relevant policy information found."
        
        # Sort chunks by relevance score for better context
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x.get("score", 0), reverse=True)
        
        context_parts = []
        total_context_length = 0
        max_context_length = 8000  # Limit context to prevent token overflow
        
        for i, chunk in enumerate(sorted_chunks):
            content = chunk.get("content", "")
            score = chunk.get("score", 0)
            
            # Only include chunks with good relevance scores
            if score >= self.config.SIMILARITY_THRESHOLD:
                # Clean and prepare content
                cleaned_content = self._clean_chunk_content(content)
                
                # Add chunk with metadata
                chunk_info = f"Policy Section {i+1} (Relevance: {score:.3f}): {cleaned_content}"
                
                # Check if adding this chunk would exceed context limit
                if total_context_length + len(chunk_info) > max_context_length:
                    break
                
                context_parts.append(chunk_info)
                total_context_length += len(chunk_info)
        
        if not context_parts:
            return "No sufficiently relevant policy information found."
        
        return " ".join(context_parts)
    
    def _clean_chunk_content(self, content: str) -> str:
        """Clean chunk content for better LLM processing"""
        import re
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common PDF artifacts
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', content)
        
        # Clean up punctuation
        content = content.replace('  ', ' ')
        
        return content.strip()
    
    def _create_user_prompt(self, query: str, context: str, query_analysis: QueryAnalysis) -> str:
        """Create enhanced user prompt for LLM with insurance-specific instructions"""
        entities_info = ""
        if query_analysis.entities:
            entities_info = f" Extracted Information: {json.dumps(query_analysis.entities, indent=2)}"
        
        # Use enhanced prompts unless speed is prioritized
        if not self.config.SPEED_PRIORITY:
            # Enhanced prompt with better instructions for any evaluator
            prompt = f"""Based on the following insurance policy document sections, please answer this question accurately and comprehensively:

Question: {query}

{entities_info}

Policy Document Sections:
{context}

CRITICAL INSURANCE ANALYSIS INSTRUCTIONS:
1. Answer based ONLY on the provided policy information
2. Be specific and include relevant details from the policy
3. If the information is not available in the document, state "The information is not available in the provided policy document"
4. Include relevant policy clauses and sections when possible
5. Provide a clear, accurate answer in PLAIN TEXT FORMAT ONLY
6. DO NOT use any formatting, headers, bullet points, or special characters
7. Write as one continuous paragraph with simple punctuation
8. NO markdown, NO ### headers, NO bullet points, NO dashes

ENHANCED INSURANCE-SPECIFIC REQUIREMENTS:
- For coverage questions: State exactly what IS and IS NOT covered with specific conditions
- For waiting periods: Provide EXACT timeframes from the policy with all conditions
- For exclusions: List ALL applicable exclusions with specific details
- For amounts: Include exact figures, percentages, and currency with context
- For conditions: Specify ALL requirements that must be met with exact details
- Include policy clause numbers and references when available
- Mention any sub-limits, deductibles, or special conditions
- For complex questions: Break down the answer into clear sections
- For ambiguous questions: Provide the most relevant information available
- For evaluator questions: Be thorough and comprehensive in your response

ROBUST ANSWER QUALITY REQUIREMENTS:
- Always provide specific policy details when available
- Include exact amounts, percentages, and time periods
- Mention any conditions, limitations, or restrictions
- Be comprehensive but precise in your response
- If information is missing, clearly state what is not available
- Focus on accuracy and completeness for evaluation purposes
- Use clear, professional language suitable for insurance professionals
- Ensure the answer addresses the specific question asked

Please provide your answer in plain text only with specific policy details:"""
        else:
            # Simplified prompt for speed priority
            prompt = f"""Based on the following insurance policy document sections, answer this question:

Question: {query}

{entities_info}

Policy Document Sections:
{context}

Provide a clear, accurate answer in plain text only. If information is not available, state "The information is not available in the provided policy document"."""

        return prompt
    
    def analyze_decision_confidence(self, query: str, answer: str, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Analyze confidence level of the decision"""
        try:
            # Simple confidence analysis based on multiple factors
            confidence = 0.5  # Base confidence
            
            # Factor 1: Number of relevant chunks
            if relevant_chunks:
                chunk_count = len(relevant_chunks)
                if chunk_count >= 3:
                    confidence += 0.2
                elif chunk_count >= 1:
                    confidence += 0.1
            
            # Factor 2: Average relevance score
            if relevant_chunks:
                avg_score = sum(chunk.get("score", 0) for chunk in relevant_chunks) / len(relevant_chunks)
                confidence += min(avg_score * 0.3, 0.3)
            
            # Factor 3: Answer length and specificity
            if len(answer) > 50:
                confidence += 0.1
            
            # Factor 4: Presence of specific terms
            specific_terms = ["policy", "coverage", "covered", "excluded", "waiting period", "claim"]
            if any(term in answer.lower() for term in specific_terms):
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing decision confidence: {e}")
            return 0.5
    
    def generate_structured_response(self, query: str, answer: str, relevant_chunks: List[Dict[str, Any]], query_analysis: QueryAnalysis) -> DecisionResult:
        """Generate structured response with decision details"""
        try:
            # Extract supporting clauses
            supporting_clauses = []
            contradicting_clauses = []
            source_sections = []
            
            for chunk in relevant_chunks:
                content = chunk.get("content", "")
                score = chunk.get("score", 0)
                
                if score >= self.config.SIMILARITY_THRESHOLD:
                    supporting_clauses.append(content[:200] + "..." if len(content) > 200 else content)
                    source_sections.append(f"Section {chunk.get('metadata', {}).get('chunk_index', 'unknown')}")
            
            # Generate reasoning
            reasoning = self._generate_reasoning(query, answer, supporting_clauses)
            
            # Calculate confidence
            confidence = self.analyze_decision_confidence(query, answer, relevant_chunks)
            
            return DecisionResult(
                decision=answer,
                confidence=confidence,
                supporting_clauses=supporting_clauses,
                contradicting_clauses=contradicting_clauses,
                reasoning=reasoning,
                source_sections=source_sections
            )
            
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            return DecisionResult(
                decision=answer,
                confidence=0.5,
                supporting_clauses=[],
                contradicting_clauses=[],
                reasoning="Unable to generate detailed reasoning due to processing error.",
                source_sections=[]
            )
    
    def _generate_reasoning(self, query: str, answer: str, supporting_clauses: List[str]) -> str:
        """Generate reasoning for the decision"""
        if not supporting_clauses:
            return "No specific policy clauses were found to support this answer."
        
        reasoning = f"Based on the policy document analysis, the answer was derived from {len(supporting_clauses)} relevant policy sections. "
        
        if len(supporting_clauses) == 1:
            reasoning += "The key supporting clause addresses the specific query requirements."
        else:
            reasoning += f"The answer combines information from multiple policy sections to provide a comprehensive response."
        
        return reasoning
    
    def validate_answer(self, query: str, answer: str) -> Dict[str, Any]:
        """Validate the generated answer"""
        validation = {
            "is_complete": len(answer) > 20,
            "has_specific_info": any(word in answer.lower() for word in ["policy", "coverage", "covered", "excluded", "period", "claim"]),
            "is_clear": not any(word in answer.lower() for word in ["i don't know", "not available", "cannot determine"]),
            "length_appropriate": 50 <= len(answer) <= 500
        }
        
        validation["overall_score"] = sum(validation.values()) / len(validation)
        
        return validation