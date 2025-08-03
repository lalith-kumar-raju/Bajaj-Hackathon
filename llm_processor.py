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
        
        # System prompts for different query types
        self.system_prompts = {
            "coverage_check": """You are an expert insurance policy analyzer. Your task is to analyze insurance policy documents and answer questions about coverage.

Key Guidelines:
1. Only base your answers on the provided policy document content
2. Be specific and cite exact clauses when possible
3. If information is not in the document, state this clearly
4. Provide clear yes/no answers when appropriate
5. Include relevant conditions and limitations
6. Use professional insurance terminology

CRITICAL FORMATTING RULES - FOLLOW EXACTLY:
- Write in PLAIN TEXT ONLY - like a simple text message
- NO markdown formatting AT ALL
- NO bold, NO italic, NO special characters
- NO escape characters like \\n or \\"
- NO line breaks or paragraph breaks
- NO bullet points or lists
- NO quotes with escape characters
- Write everything as ONE continuous paragraph
- Use simple periods and commas only
- Write as if you're sending a basic SMS text

ANSWER LENGTH RULES:
- Keep answers CONCISE and DIRECT
- Answer the question in 1-2 sentences maximum
- NO verbose explanations
- NO "information not found" messages
- If you can't answer, state this clearly
- Focus on the specific information requested

IMPORTANT: Your response must be completely plain text with no formatting whatsoever.""",

            "waiting_period": """You are an expert insurance policy analyzer specializing in waiting periods and time-based conditions.

Key Guidelines:
1. Identify specific waiting periods mentioned in the policy
2. Distinguish between different types of waiting periods (pre-existing, specific procedures, etc.)
3. Provide exact timeframes when available
4. Explain any conditions that affect waiting periods
5. Be precise about when coverage begins

CRITICAL FORMATTING RULES - FOLLOW EXACTLY:
- Write in PLAIN TEXT ONLY - like a simple text message
- NO markdown formatting AT ALL
- NO bold, NO italic, NO special characters
- NO escape characters like \\n or \\"
- NO line breaks or paragraph breaks
- NO bullet points or lists
- NO quotes with escape characters
- Write everything as ONE continuous paragraph
- Use simple periods and commas only
- Write as if you're sending a basic SMS text

ANSWER LENGTH RULES:
- Keep answers CONCISE and DIRECT
- Answer the question in 1-2 sentences maximum
- NO verbose explanations
- NO "information not found" messages
- If you can't answer, state this clearly
- Focus on the specific information requested

IMPORTANT: Your response must be completely plain text with no formatting whatsoever.""",

            "exclusion_check": """You are an expert insurance policy analyzer focusing on exclusions and limitations.

Key Guidelines:
1. Identify what is NOT covered under the policy
2. List specific exclusions and limitations
3. Explain conditions that void coverage
4. Be thorough in identifying restrictions
5. Distinguish between general exclusions and specific procedure exclusions

CRITICAL FORMATTING RULES - FOLLOW EXACTLY:
- Write in PLAIN TEXT ONLY - like a simple text message
- NO markdown formatting AT ALL
- NO bold, NO italic, NO special characters
- NO escape characters like \\n or \\"
- NO line breaks or paragraph breaks
- NO bullet points or lists
- NO quotes with escape characters
- Write everything as ONE continuous paragraph
- Use simple periods and commas only
- Write as if you're sending a basic SMS text

ANSWER LENGTH RULES:
- Keep answers CONCISE and DIRECT
- Answer the question in 1-2 sentences maximum
- NO verbose explanations
- NO "information not found" messages
- If you can't answer, state this clearly
- Focus on the specific information requested

IMPORTANT: Your response must be completely plain text with no formatting whatsoever.""",

            "policy_details": """You are an expert insurance policy analyzer providing detailed policy information.

Key Guidelines:
1. Explain policy features and benefits clearly
2. Provide specific details about coverage limits
3. Explain policy terms and conditions
4. Include relevant definitions and clarifications
5. Be comprehensive but concise

CRITICAL FORMATTING RULES - FOLLOW EXACTLY:
- Write in PLAIN TEXT ONLY - like a simple text message
- NO markdown formatting AT ALL
- NO bold, NO italic, NO special characters
- NO escape characters like \\n or \\"
- NO line breaks or paragraph breaks
- NO bullet points or lists
- NO quotes with escape characters
- Write everything as ONE continuous paragraph
- Use simple periods and commas only
- Write as if you're sending a basic SMS text

ANSWER LENGTH RULES:
- Keep answers CONCISE and DIRECT
- Answer the question in 1-2 sentences maximum
- NO verbose explanations
- NO "information not found" messages
- If you can't answer, state this clearly
- Focus on the specific information requested

IMPORTANT: Your response must be completely plain text with no formatting whatsoever.""",

            "general_query": """You are an expert insurance policy analyzer. Answer questions about the insurance policy based on the provided document content.

Key Guidelines:
1. Base answers only on the provided policy document
2. Be accurate and specific
3. Cite relevant policy sections when possible
4. Provide clear, helpful information
5. If information is not available, state this clearly

CRITICAL FORMATTING RULES - FOLLOW EXACTLY:
- Write in PLAIN TEXT ONLY - like a simple text message
- NO markdown formatting AT ALL
- NO bold, NO italic, NO special characters
- NO escape characters like \\n or \\"
- NO line breaks or paragraph breaks
- NO bullet points or lists
- NO quotes with escape characters
- Write everything as ONE continuous paragraph
- Use simple periods and commas only
- Write as if you're sending a basic SMS text

ANSWER LENGTH RULES:
- Keep answers CONCISE and DIRECT
- Answer the question in 1-2 sentences maximum
- NO verbose explanations
- NO "information not found" messages
- If you can't answer, state this clearly
- Focus on the specific information requested

IMPORTANT: Your response must be completely plain text with no formatting whatsoever."""
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
        """Simple cleaning to remove any formatting"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Remove code blocks
        
        # Remove ALL newlines and replace with spaces
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove any remaining escape characters
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\"', '"')
        
        return text.strip()
    
    def _prepare_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from relevant document chunks"""
        if not relevant_chunks:
            return "No relevant policy information found."
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            content = chunk.get("content", "")
            score = chunk.get("score", 0)
            
            # Only include chunks with good relevance scores
            if score >= self.config.SIMILARITY_THRESHOLD:
                context_parts.append(f"Policy Section {i} (Relevance: {score:.2f}): {content}")
        
        return " ".join(context_parts)
    
    def _create_user_prompt(self, query: str, context: str, query_analysis: QueryAnalysis) -> str:
        """Create user prompt for LLM"""
        entities_info = ""
        if query_analysis.entities:
            entities_info = f" Extracted Information: {json.dumps(query_analysis.entities, indent=2)}"
        
        prompt = f"""Based on the following insurance policy document sections, please answer this question:

Question: {query}

{entities_info}

Policy Document Sections:
{context}

Please provide a clear, accurate answer based only on the policy information provided above. If the information is not available in the document, please state this clearly."""

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