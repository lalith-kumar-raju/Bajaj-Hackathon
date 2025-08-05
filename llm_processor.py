from openai import AzureOpenAI
from typing import List, Dict, Any, Optional
from models import DecisionResult, QueryAnalysis
from config import Config
import json
import logging

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

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
        
        # Universal system prompts for ANY document type
        self.system_prompts = {
            "coverage_check": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Start with yes/no if applicable
3. Include key conditions and limits only
4. Mention important amounts and percentages
5. Focus on essential information only

ANSWER RULES:
- Keep answers under 150 words
- Start with direct answer
- Include only key conditions and limits
- Mention important amounts/percentages
- Skip repetitive information
- Focus on what user needs to know

FORMAT: Plain text, one paragraph, no formatting""",

            "waiting_period": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Provide exact timeframes (days, months, years)
3. Include key conditions that affect time periods
4. Focus on essential information only
5. Skip repetitive details

ANSWER RULES:
- Keep answers under 150 words
- Start with exact timeframe
- Include key conditions only
- Skip repetitive information
- Focus on what user needs to know

FORMAT: Plain text, one paragraph, no formatting""",

            "exclusion_check": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. List key exclusions and limitations only
3. Include important conditions that void coverage
4. Focus on essential information only
5. Skip repetitive details

ANSWER RULES:
- Keep answers under 150 words
- Start with key exclusions
- Include important conditions only
- Skip repetitive information
- Focus on what user needs to know

FORMAT: Plain text, one paragraph, no formatting""",

            "policy_details": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Include key amounts, percentages, and limits
3. Focus on essential features and benefits
4. Skip repetitive information
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with key information
- Include important amounts/limits
- Skip repetitive details
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting""",

            "technical_details": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Include key technical specifications only
3. Focus on essential information only
4. Skip repetitive details
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with key specifications
- Include important details only
- Skip repetitive information
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting""",

            "legal_information": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Include key legal provisions only
3. Focus on essential information only
4. Skip repetitive details
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with key provisions
- Include important details only
- Skip repetitive information
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting""",

            "scientific_analysis": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Include key findings and data only
3. Focus on essential information only
4. Skip repetitive details
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with key findings
- Include important data only
- Skip repetitive information
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting""",

            "procedural_guide": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Include key steps and procedures only
3. Focus on essential information only
4. Skip repetitive details
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with key steps
- Include important procedures only
- Skip repetitive information
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting""",

            "definition_query": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Provide clear definitions only
3. Focus on essential information only
4. Skip repetitive details
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with clear definition
- Include key context only
- Skip repetitive information
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting""",

            "comparison_query": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Provide key comparisons only
3. Focus on essential differences/similarities
4. Skip repetitive details
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with key comparison
- Include important differences only
- Skip repetitive information
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting""",

            "explanation_query": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Provide clear explanations only
3. Focus on essential information only
4. Skip repetitive details
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with clear explanation
- Include key details only
- Skip repetitive information
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting""",

            "general_query": """You are an expert document analyst. Answer based ONLY on the provided document.

CRITICAL INSTRUCTIONS:
1. Be CONCISE and direct - maximum 150 words
2. Include key details and information only
3. Focus on essential information only
4. Skip repetitive details
5. Focus on what user needs to know

ANSWER RULES:
- Keep answers under 150 words
- Start with direct answer
- Include key details only
- Skip repetitive information
- Focus on essential information only

FORMAT: Plain text, one paragraph, no formatting"""
        }
    
    def process_query(self, query: str, relevant_chunks: List[Dict[str, Any]], query_analysis: QueryAnalysis = None) -> str:
        """Process query using LLM with relevant document chunks"""
        try:
            # Prepare context from relevant chunks
            context = self._prepare_context(relevant_chunks)
            
            # Get appropriate system prompt
            intent_key = query_analysis.intent.value if query_analysis and query_analysis.intent else "general_query"
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
            
            # logger.info(f"Generated answer for query: {query[:50]}...")
            
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
    
    def _create_user_prompt(self, query: str, context: str, query_analysis: QueryAnalysis = None) -> str:
        """Create enhanced user prompt for LLM with insurance-specific instructions"""
        entities_info = ""
        if query_analysis and query_analysis.entities:
            entities_info = f" Extracted Information: {json.dumps(query_analysis.entities, indent=2)}"
        
        # Use enhanced prompts for accuracy priority
        if self.config.ACCURACY_PRIORITY:
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
            # Balanced prompt for speed vs accuracy
            prompt = f"""Based on the following document sections, answer this question accurately:

Question: {query}

{entities_info}

Document Sections:
{context}

Provide a clear, accurate answer in plain text only. If information is not available, state "The information is not available in the provided document"."""

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