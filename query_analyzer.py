import re
import spacy
from typing import Dict, Any, List, Optional
from models import QueryAnalysis, QueryIntent
from config import Config
import logging

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(self):
        self.config = Config()
        # Load spaCy model for NER (fallback to English if not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not available, using basic NER")
            self.nlp = None
        
        # Universal insurance keywords for ANY question type
        self.intent_keywords = {
            QueryIntent.COVERAGE_CHECK: [
                "cover", "coverage", "covered", "include", "policy covers", "eligible", "pay", "reimburse",
                "surgery", "treatment", "procedure", "medical", "hospitalization", "benefit", "provide",
                "what is covered", "what does it cover", "what's included", "what's covered"
            ],
            QueryIntent.WAITING_PERIOD: [
                "waiting period", "wait", "time", "duration", "months", "years", "days",
                "pre-existing", "existing condition", "initial period", "when", "how long",
                "timeframe", "period", "delay", "before coverage"
            ],
            QueryIntent.EXCLUSION_CHECK: [
                "exclude", "exclusion", "not covered", "not include", "restriction", "limitation",
                "not applicable", "what's not covered", "what's excluded", "what's not included",
                "limitations", "restrictions", "not eligible", "not covered under"
            ],
            QueryIntent.CLAIM_PROCESS: [
                "claim", "claim process", "how to claim", "claim procedure", "documentation",
                "submit claim", "claim form", "how to file", "claiming", "reimbursement",
                "claim submission", "claim documentation", "claim requirements"
            ],
            QueryIntent.POLICY_DETAILS: [
                "policy", "details", "information", "what is", "define", "explain", "describe",
                "tell me about", "policy terms", "policy conditions", "policy features",
                "policy benefits", "policy limits", "policy coverage"
            ]
        }
        
        # Universal insurance entities for ANY question
        self.insurance_entities = {
            "age": r"(\d+)\s*(?:year|yr)s?\s*old",
            "procedure": r"(surgery|treatment|procedure|operation)\s*(?:for|of)?\s*([^,\.]+)",
            "location": r"(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            "policy_duration": r"(\d+)\s*(?:month|year)s?\s*(?:old\s+)?policy",
            "amount": r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹|percent|%)",
            "disease": r"(?:for|of|with)\s+([^,\.]+(?:\s+disease|\s+condition))",
            "hospital": r"(?:at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:hospital|clinic|medical center)",
            "insurance_type": r"(?:health|medical|life|accident|travel)\s+insurance",
            "coverage_type": r"(?:inpatient|outpatient|day care|pre|post)\s+(?:treatment|surgery|care)",
            "time_period": r"(\d+)\s*(?:days?|months?|years?)",
            "percentage": r"(\d+)\s*(?:percent|%)",
            "currency": r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)",
            "medical_term": r"(?:surgery|treatment|procedure|operation|therapy|medication|diagnosis)",
            "condition": r"(?:pre-existing|existing|chronic|acute|temporary|permanent)\s+(?:condition|disease|illness)"
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Universal query analysis for ANY type of question"""
        query_lower = query.lower().strip()
        
        # Universal intent classification
        intent = self._classify_universal_intent(query_lower)
        
        # Universal entity extraction
        entities = self._extract_universal_entities(query)
        
        # Universal confidence calculation
        confidence = self._calculate_universal_confidence(query_lower, intent)
        
        # Universal query processing
        processed_query = self._process_universal_query(query, intent, entities)
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            confidence=confidence,
            processed_query=processed_query
        )
    
    def _classify_universal_intent(self, query: str) -> QueryIntent:
        """Universal intent classification for ANY question type"""
        intent_scores = {}
        
        # Enhanced scoring for better accuracy
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                # Exact match gets higher score
                if keyword in query:
                    score += 2
                # Partial match gets lower score
                elif any(word in query for word in keyword.split()):
                    score += 1
            intent_scores[intent] = score
        
        # Find intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        # Universal fallback - analyze question type
        return self._analyze_question_type(query)
    
    def _analyze_question_type(self, query: str) -> QueryIntent:
        """Analyze question type for universal classification"""
        query_lower = query.lower()
        
        # Question word analysis
        question_words = ["what", "how", "when", "where", "why", "which", "who"]
        question_word = next((word for word in question_words if word in query_lower), None)
        
        # Action word analysis
        action_words = {
            "cover": QueryIntent.COVERAGE_CHECK,
            "wait": QueryIntent.WAITING_PERIOD,
            "exclude": QueryIntent.EXCLUSION_CHECK,
            "claim": QueryIntent.CLAIM_PROCESS,
            "define": QueryIntent.POLICY_DETAILS,
            "explain": QueryIntent.POLICY_DETAILS,
            "describe": QueryIntent.POLICY_DETAILS
        }
        
        for action, intent in action_words.items():
            if action in query_lower:
                return intent
        
        # Default to general query for unknown types
        return QueryIntent.GENERAL_QUERY
    
    def _extract_universal_entities(self, query: str) -> Dict[str, Any]:
        """Universal entity extraction for ANY question"""
        entities = {}
        
        # Extract using enhanced regex patterns
        for entity_type, pattern in self.insurance_entities.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Use spaCy for additional NER if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "MONEY", "QUANTITY"]:
                    entities[f"ner_{ent.label_.lower()}"] = ent.text
        
        # Universal entity extraction
        entities.update(self._extract_universal_patterns(query))
        
        return entities
    
    def _extract_universal_patterns(self, query: str) -> Dict[str, Any]:
        """Extract universal patterns from ANY question"""
        patterns = {}
        
        # Extract numbers
        numbers = re.findall(r'\d+', query)
        if numbers:
            patterns["numbers"] = numbers
        
        # Extract amounts
        amounts = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)', query, re.IGNORECASE)
        if amounts:
            patterns["amounts"] = amounts
        
        # Extract time periods
        time_periods = re.findall(r'(\d+)\s*(?:days?|months?|years?)', query, re.IGNORECASE)
        if time_periods:
            patterns["time_periods"] = time_periods
        
        # Extract medical terms
        medical_terms = re.findall(r'(surgery|treatment|procedure|operation|therapy|medication|diagnosis)', query, re.IGNORECASE)
        if medical_terms:
            patterns["medical_terms"] = medical_terms
        
        # Extract conditions
        conditions = re.findall(r'(pre-existing|existing|chronic|acute|temporary|permanent)', query, re.IGNORECASE)
        if conditions:
            patterns["conditions"] = conditions
        
        return patterns
    
    def _calculate_universal_confidence(self, query: str, intent: QueryIntent) -> float:
        """Universal confidence calculation for ANY question"""
        if intent == QueryIntent.GENERAL_QUERY:
            return 0.7  # Higher base confidence for general queries
        
        keywords = self.intent_keywords.get(intent, [])
        if not keywords:
            return 0.7
        
        matches = sum(1 for keyword in keywords if keyword in query)
        confidence = min(matches / len(keywords), 1.0)
        
        # Boost confidence for specific patterns
        if intent == QueryIntent.COVERAGE_CHECK and any(word in query for word in ["cover", "coverage", "covered"]):
            confidence = min(confidence + 0.2, 1.0)
        elif intent == QueryIntent.WAITING_PERIOD and "waiting period" in query:
            confidence = min(confidence + 0.3, 1.0)
        elif intent == QueryIntent.EXCLUSION_CHECK and "exclusion" in query:
            confidence = min(confidence + 0.2, 1.0)
        
        return confidence
    
    def _process_universal_query(self, query: str, intent: QueryIntent, entities: Dict[str, Any]) -> str:
        """Universal query processing for ANY question type"""
        processed = query
        
        # Add universal context for better retrieval
        if intent == QueryIntent.COVERAGE_CHECK:
            if "cover" not in processed.lower():
                processed += " coverage policy benefits"
        elif intent == QueryIntent.WAITING_PERIOD:
            if "waiting period" not in processed.lower():
                processed += " waiting period time duration"
        elif intent == QueryIntent.EXCLUSION_CHECK:
            if "exclusion" not in processed.lower():
                processed += " exclusion limitation restriction"
        elif intent == QueryIntent.CLAIM_PROCESS:
            if "claim" not in processed.lower():
                processed += " claim process procedure"
        elif intent == QueryIntent.POLICY_DETAILS:
            if "policy" not in processed.lower():
                processed += " policy terms conditions"
        
        # Add universal entity information
        if entities.get("medical_terms"):
            processed += f" {' '.join(entities['medical_terms'])}"
        if entities.get("time_periods"):
            processed += f" {' '.join(entities['time_periods'])}"
        if entities.get("amounts"):
            processed += f" {' '.join(entities['amounts'])}"
        
        return processed.strip()
    
    def get_universal_suggestions(self, query: str) -> List[str]:
        """Generate universal query suggestions for ANY question"""
        suggestions = []
        
        # Add variations with different terms
        base_query = query.lower()
        
        # Universal synonym expansion
        synonyms = {
            "cover": ["coverage", "include", "provide", "offer"],
            "waiting period": ["wait", "time", "duration", "period"],
            "exclude": ["exclusion", "not covered", "restriction", "limitation"],
            "claim": ["claim process", "claim procedure", "claim submission"],
            "policy": ["insurance", "coverage", "plan", "terms"],
            "surgery": ["operation", "procedure", "treatment"],
            "hospital": ["medical center", "clinic", "healthcare facility"],
            "benefit": ["coverage", "advantage", "feature", "provision"]
        }
        
        for original, syns in synonyms.items():
            if original in base_query:
                for syn in syns:
                    suggestions.append(query.replace(original, syn))
        
        # Add question variations
        question_variations = [
            query.replace("what is", "how does"),
            query.replace("how", "what"),
            query.replace("when", "what time"),
            query.replace("where", "in what location")
        ]
        suggestions.extend(question_variations)
        
        return list(set(suggestions))[:8]  # Limit to 8 unique suggestions