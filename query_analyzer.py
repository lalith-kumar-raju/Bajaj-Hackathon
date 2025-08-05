import re
import spacy
from typing import Dict, Any, List, Optional
from models import QueryAnalysis, QueryIntent
from config import Config
import logging
import hashlib

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
        
        # Enhanced universal keywords for ANY document type and question
        self.intent_keywords = {
            QueryIntent.COVERAGE_CHECK: [
                "cover", "coverage", "covered", "include", "policy covers", "eligible", "pay", "reimburse",
                "surgery", "treatment", "procedure", "medical", "hospitalization", "benefit", "provide",
                "what is covered", "what does it cover", "what's included", "what's covered", "does this cover",
                "is this covered", "will it cover", "coverage for", "benefits for", "what benefits",
                "what's the coverage", "what does the policy cover", "what are the benefits"
            ],
            QueryIntent.WAITING_PERIOD: [
                "waiting period", "wait", "time", "duration", "months", "years", "days",
                "pre-existing", "existing condition", "initial period", "when", "how long",
                "timeframe", "period", "delay", "before coverage", "waiting time", "waiting duration",
                "how many days", "how many months", "how many years", "time limit", "time restriction",
                "when does coverage start", "when will it be covered", "waiting period for"
            ],
            QueryIntent.EXCLUSION_CHECK: [
                "exclude", "exclusion", "not covered", "not include", "restriction", "limitation",
                "not applicable", "what's not covered", "what's excluded", "what's not included",
                "limitations", "restrictions", "not eligible", "not covered under", "what's excluded",
                "what's not covered", "what's not included", "what's not eligible", "what's restricted",
                "what's limited", "what's not applicable", "what's not allowed", "what's not permitted"
            ],
            QueryIntent.CLAIM_PROCESS: [
                "claim", "claim process", "how to claim", "claim procedure", "documentation",
                "submit claim", "claim form", "how to file", "claiming", "reimbursement",
                "claim submission", "claim documentation", "claim requirements", "how to submit",
                "how to apply", "how to file claim", "claim process", "claiming process",
                "reimbursement process", "claim procedure", "claim documentation", "claim requirements"
            ],
            QueryIntent.POLICY_DETAILS: [
                "policy", "details", "information", "what is", "define", "explain", "describe",
                "tell me about", "policy terms", "policy conditions", "policy features",
                "policy benefits", "policy limits", "policy coverage", "what is the policy",
                "policy information", "policy details", "policy terms and conditions",
                "policy features", "policy benefits", "policy coverage", "policy limits"
            ],
            QueryIntent.TECHNICAL_DETAILS: [
                "technical", "specification", "spec", "specs", "technical details", "technical information",
                "specifications", "technical specs", "technical data", "technical specifications",
                "what are the specifications", "what are the specs", "technical details", "technical info",
                "specification details", "technical specifications", "technical data", "technical information"
            ],
            QueryIntent.LEGAL_INFORMATION: [
                "legal", "law", "legal information", "legal details", "legal provisions", "legal requirements",
                "legal terms", "legal conditions", "legal rights", "legal duties", "legal obligations",
                "what are the legal provisions", "what are the legal requirements", "legal information",
                "legal details", "legal terms", "legal conditions", "legal rights", "legal duties"
            ],
            QueryIntent.SCIENTIFIC_ANALYSIS: [
                "scientific", "research", "study", "analysis", "scientific analysis", "scientific information",
                "scientific details", "scientific data", "scientific results", "scientific findings",
                "what are the scientific findings", "what are the research results", "scientific analysis",
                "scientific information", "scientific details", "scientific data", "scientific results"
            ],
            QueryIntent.PROCEDURAL_GUIDE: [
                "procedure", "process", "step", "instruction", "guide", "how to", "procedural guide",
                "procedural instructions", "step by step", "procedure steps", "process steps",
                "what are the steps", "how to do", "procedure guide", "process guide", "instruction guide",
                "step by step guide", "procedural instructions", "procedure steps", "process steps"
            ],
            QueryIntent.DEFINITION_QUERY: [
                "define", "definition", "what is", "what does", "meaning", "explain", "describe",
                "define the term", "what is the definition", "what does it mean", "explain the meaning",
                "what is the meaning", "definition of", "meaning of", "explain what", "describe what"
            ],
            QueryIntent.COMPARISON_QUERY: [
                "compare", "comparison", "difference", "similar", "different", "versus", "vs",
                "what is the difference", "how are they different", "compare and contrast",
                "what are the differences", "similarities and differences", "compare the",
                "difference between", "similar to", "different from", "versus", "vs"
            ],
            QueryIntent.EXPLANATION_QUERY: [
                "explain", "explanation", "how", "why", "what", "when", "where", "explain how",
                "explain why", "explain what", "explain when", "explain where", "how does",
                "why does", "what does", "when does", "where does", "explanation of", "explain the"
            ]
        }
        
        # Enhanced universal insurance entities for ANY question
        self.insurance_entities = {
            "age": r"(\d+)\s*(?:year|yr)s?\s*old",
            "procedure": r"(surgery|treatment|procedure|operation)\s*(?:for|of)?\s*([^,\.]+)",
            "location": r"(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            "policy_duration": r"(\d+)\s*(?:month|year)s?\s*(?:old\s+)?policy",
            "amount": r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakhs?|crores?|thousand|rs|rupees|inr|₹|percent|%)",
            "amount_words": r"(fifty|sixty|seventy|eighty|ninety|hundred|thousand|lakh|crore)\s*(?:rupees|rs|₹)?",
            "policy_number": r"(?:policy\s*(?:no\.?|number)\s*:?\s*)([A-Z0-9\-\/]+)",
            "percentage": r"(\d+(?:\.\d+)?)\s*(?:percent|%|per\s*cent)",
            "disease": r"(?:for|of|with)\s+([^,\.]+(?:\s+disease|\s+condition))",
            "hospital": r"(?:at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:hospital|clinic|medical center)",
            "insurance_type": r"(?:health|medical|life|accident|travel)\s+insurance",
            "coverage_type": r"(?:inpatient|outpatient|day care|pre|post)\s+(?:treatment|surgery|care)",
            "time_period": r"(\d+)\s*(?:days?|months?|years?)",
            "currency": r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)",
            "medical_term": r"(?:surgery|treatment|procedure|operation|therapy|medication|diagnosis)",
            "condition": r"(?:pre-existing|existing|chronic|acute|temporary|permanent)\s+(?:condition|disease|illness)",
            "waiting_period": r"(?:waiting period|wait)\s*(?:of|for)?\s*(\d+)\s*(?:days?|months?|years?)",
            "grace_period": r"(?:grace period|grace)\s*(?:of|for)?\s*(\d+)\s*(?:days?|months?|years?)",
            "sum_insured": r"(?:sum insured|coverage|limit)\s*(?:of|up to)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakhs?|crores?|thousand|rs|rupees|inr|₹)?",
            "deductible": r"(?:deductible|excess|co-payment)\s*(?:of|up to)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)?",
            "exclusion": r"(?:exclusion|excluded|not covered|not include)\s*(?:for|of)?\s*([^,\.]+)",
            "benefit": r"(?:benefit|coverage|include)\s*(?:for|of)?\s*([^,\.]+)",
            "policy_name": r"(?:policy|plan)\s+(?:name|called|titled)\s+([^,\.]+)",
            "premium": r"(?:premium|payment|amount)\s*(?:of|is)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)?",
            "coverage_limit": r"(?:coverage|limit|maximum)\s*(?:of|up to)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)?",
            "room_rent": r"(?:room|accommodation)\s+(?:rent|charge|cost)\s*(?:limit|maximum)?",
            "icu_charges": r"(?:icu|intensive care)\s+(?:charges|cost|expenses)",
            "pre_hospitalization": r"(?:pre|before)\s+(?:hospitalization|admission)",
            "post_hospitalization": r"(?:post|after)\s+(?:hospitalization|discharge)",
            "day_care": r"(?:day care|daycare|same day)\s+(?:surgery|procedure|treatment)",
            "organ_transplant": r"(?:organ|kidney|liver|heart)\s+(?:transplant|donor)",
            "maternity": r"(?:maternity|pregnancy|delivery|childbirth|cesarean)",
            "dental": r"(?:dental|tooth|teeth|oral)\s+(?:treatment|surgery|care)",
            "ophthalmic": r"(?:eye|ophthalmic|ophthalmology|vision)\s+(?:treatment|surgery)",
            "mental_health": r"(?:mental|psychiatric|psychological|psychology)\s+(?:health|treatment|care)",
            "ayush": r"(?:ayush|ayurveda|homeopathy|yoga|naturopathy|siddha|unani)",
            "preventive": r"(?:preventive|prevention|health check|checkup|screening)",
            "emergency": r"(?:emergency|urgent|critical)\s+(?:care|treatment|admission)",
            "ambulance": r"(?:ambulance|emergency transport|medical transport)",
            "dialysis": r"(?:dialysis|kidney dialysis|renal dialysis)",
            "chemotherapy": r"(?:chemotherapy|chemo|cancer treatment)",
            "radiotherapy": r"(?:radiotherapy|radiation|radiation therapy)",
            "prosthesis": r"(?:prosthesis|artificial limb|artificial organ)",
            "implants": r"(?:implant|artificial|prosthetic)\s+(?:device|organ|limb)"
        }
        
        # Enhanced question patterns for better classification
        self.question_patterns = {
            "what": ["what is", "what are", "what does", "what will", "what can", "what should"],
            "how": ["how much", "how many", "how long", "how does", "how to", "how will"],
            "when": ["when does", "when will", "when can", "when should", "when is"],
            "where": ["where can", "where does", "where will", "where is"],
            "why": ["why is", "why does", "why will", "why can"],
            "which": ["which", "which one", "which type", "which kind"],
            "is": ["is there", "is this", "is it", "is the"],
            "does": ["does this", "does it", "does the", "does that"],
            "will": ["will this", "will it", "will the", "will that"],
            "can": ["can this", "can it", "can the", "can that"]
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Enhanced universal query analysis for ANY type of question"""
        query_lower = query.lower().strip()
        
        # Enhanced intent classification with fallback
        intent = self._classify_enhanced_intent(query_lower)
        
        # Enhanced entity extraction
        entities = self._extract_enhanced_entities(query)
        
        # Enhanced confidence calculation
        confidence = self._calculate_enhanced_confidence(query_lower, intent, entities)
        
        # Enhanced query processing with synonyms
        processed_query = self._process_enhanced_query(query, intent, entities)
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            confidence=confidence,
            processed_query=processed_query
        )
    
    def _classify_enhanced_intent(self, query: str) -> QueryIntent:
        """Enhanced intent classification with multiple fallback strategies"""
        intent_scores = {}
        
        # Strategy 1: Keyword-based scoring
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                # Exact match gets higher score
                if keyword in query:
                    score += 3
                # Partial match gets lower score
                elif any(word in query for word in keyword.split()):
                    score += 1
            intent_scores[intent] = score
        
        # Strategy 2: Question pattern analysis
        question_score = self._analyze_question_patterns(query)
        if question_score:
            intent_scores[question_score] = intent_scores.get(question_score, 0) + 2
        
        # Strategy 3: Action word analysis
        action_score = self._analyze_action_words(query)
        if action_score:
            intent_scores[action_score] = intent_scores.get(action_score, 0) + 1
        
        # Find intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        # Strategy 4: Universal fallback - analyze question type
        return self._analyze_question_type_enhanced(query)
    
    def _analyze_question_patterns(self, query: str) -> Optional[QueryIntent]:
        """Analyze question patterns for intent classification"""
        query_lower = query.lower()
        
        # Check for specific question patterns
        for pattern_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    # Map patterns to intents
                    if pattern_type in ["what", "which"]:
                        if any(word in query_lower for word in ["cover", "coverage", "benefit", "include"]):
                            return QueryIntent.COVERAGE_CHECK
                        elif any(word in query_lower for word in ["wait", "period", "time", "duration"]):
                            return QueryIntent.WAITING_PERIOD
                        elif any(word in query_lower for word in ["exclude", "not", "restriction", "limitation"]):
                            return QueryIntent.EXCLUSION_CHECK
                        elif any(word in query_lower for word in ["claim", "process", "procedure", "submit"]):
                            return QueryIntent.CLAIM_PROCESS
                        else:
                            return QueryIntent.POLICY_DETAILS
                    elif pattern_type in ["how", "when"]:
                        if any(word in query_lower for word in ["claim", "submit", "process"]):
                            return QueryIntent.CLAIM_PROCESS
                        elif any(word in query_lower for word in ["wait", "period", "time"]):
                            return QueryIntent.WAITING_PERIOD
                        else:
                            return QueryIntent.POLICY_DETAILS
                    elif pattern_type in ["is", "does", "will", "can"]:
                        if any(word in query_lower for word in ["cover", "coverage", "benefit", "include"]):
                            return QueryIntent.COVERAGE_CHECK
                        elif any(word in query_lower for word in ["exclude", "not", "restriction"]):
                            return QueryIntent.EXCLUSION_CHECK
                        else:
                            return QueryIntent.POLICY_DETAILS
        
        return None
    
    def _analyze_action_words(self, query: str) -> Optional[QueryIntent]:
        """Analyze action words for intent classification"""
        query_lower = query.lower()
        
        action_words = {
            "cover": QueryIntent.COVERAGE_CHECK,
            "include": QueryIntent.COVERAGE_CHECK,
            "provide": QueryIntent.COVERAGE_CHECK,
            "benefit": QueryIntent.COVERAGE_CHECK,
            "wait": QueryIntent.WAITING_PERIOD,
            "delay": QueryIntent.WAITING_PERIOD,
            "period": QueryIntent.WAITING_PERIOD,
            "exclude": QueryIntent.EXCLUSION_CHECK,
            "restrict": QueryIntent.EXCLUSION_CHECK,
            "limit": QueryIntent.EXCLUSION_CHECK,
            "not": QueryIntent.EXCLUSION_CHECK,
            "claim": QueryIntent.CLAIM_PROCESS,
            "submit": QueryIntent.CLAIM_PROCESS,
            "file": QueryIntent.CLAIM_PROCESS,
            "process": QueryIntent.CLAIM_PROCESS,
            "define": QueryIntent.POLICY_DETAILS,
            "explain": QueryIntent.POLICY_DETAILS,
            "describe": QueryIntent.POLICY_DETAILS,
            "tell": QueryIntent.POLICY_DETAILS
        }
        
        for action, intent in action_words.items():
            if action in query_lower:
                return intent
        
        return None
    
    def _analyze_question_type_enhanced(self, query: str) -> QueryIntent:
        """Enhanced question type analysis with multiple strategies"""
        query_lower = query.lower()
        
        # Strategy 1: Question word analysis
        question_words = ["what", "how", "when", "where", "why", "which", "who"]
        question_word = next((word for word in question_words if word in query_lower), None)
        
        # Strategy 2: Coverage-related words
        coverage_words = ["cover", "coverage", "benefit", "include", "provide", "eligible"]
        if any(word in query_lower for word in coverage_words):
            return QueryIntent.COVERAGE_CHECK
        
        # Strategy 3: Time-related words
        time_words = ["wait", "period", "time", "duration", "delay", "when"]
        if any(word in query_lower for word in time_words):
            return QueryIntent.WAITING_PERIOD
        
        # Strategy 4: Exclusion-related words
        exclusion_words = ["exclude", "not", "restriction", "limitation", "not covered"]
        if any(word in query_lower for word in exclusion_words):
            return QueryIntent.EXCLUSION_CHECK
        
        # Strategy 5: Process-related words
        process_words = ["claim", "submit", "file", "process", "procedure", "how to"]
        if any(word in query_lower for word in process_words):
            return QueryIntent.CLAIM_PROCESS
        
        # Strategy 6: General information words
        info_words = ["what is", "explain", "describe", "tell me", "information", "details"]
        if any(word in query_lower for word in info_words):
            return QueryIntent.POLICY_DETAILS
        
        # Default to general query for unknown types
        return QueryIntent.GENERAL_QUERY
    
    def _extract_enhanced_entities(self, query: str) -> Dict[str, Any]:
        """Enhanced entity extraction with multiple strategies"""
        entities = {}
        
        # Strategy 1: Enhanced regex patterns
        for entity_type, pattern in self.insurance_entities.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Strategy 2: spaCy NER if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "MONEY", "QUANTITY"]:
                    entities[f"ner_{ent.label_.lower()}"] = ent.text
        
        # Strategy 3: Enhanced pattern extraction
        entities.update(self._extract_enhanced_patterns(query))
        
        # Strategy 4: Synonym expansion
        entities.update(self._extract_synonym_entities(query))
        
        return entities
    
    def _extract_enhanced_patterns(self, query: str) -> Dict[str, Any]:
        """Extract enhanced patterns from ANY question"""
        patterns = {}
        
        # Extract numbers with context
        numbers = re.findall(r'\d+', query)
        if numbers:
            patterns["numbers"] = numbers
        
        # Extract amounts with currency
        amounts = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)', query, re.IGNORECASE)
        if amounts:
            patterns["amounts"] = amounts
        
        # Extract time periods with context
        time_periods = re.findall(r'(\d+)\s*(?:days?|months?|years?)', query, re.IGNORECASE)
        if time_periods:
            patterns["time_periods"] = time_periods
        
        # Extract medical terms with context
        medical_terms = re.findall(r'(surgery|treatment|procedure|operation|therapy|medication|diagnosis)', query, re.IGNORECASE)
        if medical_terms:
            patterns["medical_terms"] = medical_terms
        
        # Extract conditions with context
        conditions = re.findall(r'(pre-existing|existing|chronic|acute|temporary|permanent)', query, re.IGNORECASE)
        if conditions:
            patterns["conditions"] = conditions
        
        # Extract policy-specific terms
        policy_terms = re.findall(r'(premium|coverage|benefit|claim|exclusion|waiting period|grace period)', query, re.IGNORECASE)
        if policy_terms:
            patterns["policy_terms"] = policy_terms
        
        return patterns
    
    def _extract_synonym_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities using synonym expansion"""
        synonyms = {}
        
        # Coverage synonyms
        coverage_synonyms = ["cover", "coverage", "include", "provide", "offer", "benefit", "advantage"]
        if any(syn in query.lower() for syn in coverage_synonyms):
            synonyms["coverage_related"] = True
        
        # Time synonyms
        time_synonyms = ["wait", "period", "time", "duration", "delay", "when", "how long"]
        if any(syn in query.lower() for syn in time_synonyms):
            synonyms["time_related"] = True
        
        # Exclusion synonyms
        exclusion_synonyms = ["exclude", "not", "restriction", "limitation", "not covered", "not included"]
        if any(syn in query.lower() for syn in exclusion_synonyms):
            synonyms["exclusion_related"] = True
        
        # Process synonyms
        process_synonyms = ["claim", "submit", "file", "process", "procedure", "how to", "apply"]
        if any(syn in query.lower() for syn in process_synonyms):
            synonyms["process_related"] = True
        
        return synonyms
    
    def _calculate_enhanced_confidence(self, query: str, intent: QueryIntent, entities: Dict[str, Any]) -> float:
        """Enhanced confidence calculation with ACCURACY priority"""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Intent keyword matches (enhanced for accuracy)
        keywords = self.intent_keywords.get(intent, [])
        if keywords:
            matches = sum(1 for keyword in keywords if keyword.lower() in query.lower())
            confidence += min(matches / len(keywords) * 0.4, 0.4)  # Increased weight
        
        # Factor 2: Entity presence (enhanced for accuracy)
        if entities:
            confidence += min(len(entities) * 0.15, 0.3)  # Increased weight
        
        # Factor 3: Question pattern analysis (enhanced)
        question_words = ["what", "how", "when", "where", "why", "which", "who", "whose"]
        if any(word in query.lower() for word in question_words):
            confidence += 0.15  # Increased weight
        
        # Factor 4: Specific entity types (enhanced for accuracy)
        if entities.get("medical_terms"):
            confidence += 0.15
        if entities.get("time_periods"):
            confidence += 0.15
        if entities.get("amounts"):
            confidence += 0.15
        if entities.get("policy_terms"):
            confidence += 0.15
        
        # Factor 5: Intent-specific boosts (enhanced for accuracy)
        if intent == QueryIntent.COVERAGE_CHECK and any(word in query.lower() for word in ["cover", "coverage", "covered", "include", "provide"]):
            confidence += 0.25
        elif intent == QueryIntent.WAITING_PERIOD and any(word in query.lower() for word in ["waiting period", "wait", "time", "duration"]):
            confidence += 0.3
        elif intent == QueryIntent.EXCLUSION_CHECK and any(word in query.lower() for word in ["exclusion", "exclude", "not covered", "restriction"]):
            confidence += 0.25
        elif intent == QueryIntent.CLAIM_PROCESS and any(word in query.lower() for word in ["claim", "file", "submit", "process"]):
            confidence += 0.25
        
        # Factor 6: Query complexity (new factor for accuracy)
        if len(query.split()) > 8:  # Complex queries
            confidence += 0.1
        
        # Factor 7: Domain-specific terms (new factor for accuracy)
        domain_terms = ["policy", "insurance", "coverage", "benefit", "premium", "claim", "exclusion"]
        if any(term in query.lower() for term in domain_terms):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _process_enhanced_query(self, query: str, intent: QueryIntent, entities: Dict[str, Any]) -> str:
        """Enhanced query processing with synonym expansion"""
        processed = query
        
        # Add intent-specific context
        if intent == QueryIntent.COVERAGE_CHECK:
            if "cover" not in processed.lower():
                processed += " coverage policy benefits include"
        elif intent == QueryIntent.WAITING_PERIOD:
            if "waiting period" not in processed.lower():
                processed += " waiting period time duration delay"
        elif intent == QueryIntent.EXCLUSION_CHECK:
            if "exclusion" not in processed.lower():
                processed += " exclusion limitation restriction not covered"
        elif intent == QueryIntent.CLAIM_PROCESS:
            if "claim" not in processed.lower():
                processed += " claim process procedure submission"
        elif intent == QueryIntent.POLICY_DETAILS:
            if "policy" not in processed.lower():
                processed += " policy terms conditions features"
        
        # Add entity information
        if entities.get("medical_terms"):
            processed += f" {' '.join(entities['medical_terms'])}"
        if entities.get("time_periods"):
            processed += f" {' '.join(entities['time_periods'])}"
        if entities.get("amounts"):
            processed += f" {' '.join(entities['amounts'])}"
        if entities.get("policy_terms"):
            processed += f" {' '.join(entities['policy_terms'])}"
        
        # Add synonym expansion
        processed = self._expand_synonyms(processed)
        
        return processed.strip()
    
    def _expand_synonyms(self, query: str) -> str:
        """Expand query with synonyms for better retrieval"""
        synonyms = {
            "cover": ["coverage", "include", "provide", "offer", "benefit"],
            "waiting period": ["wait", "time", "duration", "period", "delay"],
            "exclude": ["exclusion", "not covered", "restriction", "limitation"],
            "claim": ["claim process", "claim procedure", "claim submission", "apply"],
            "policy": ["insurance", "coverage", "plan", "terms", "conditions"],
            "surgery": ["operation", "procedure", "treatment", "surgical"],
            "hospital": ["medical center", "clinic", "healthcare facility", "hospital"],
            "benefit": ["coverage", "advantage", "feature", "provision", "benefit"],
            "premium": ["payment", "amount", "cost", "fee", "premium"],
            "coverage": ["benefit", "include", "cover", "provide", "coverage"],
            "exclusion": ["not covered", "restriction", "limitation", "exclude"],
            "waiting": ["wait", "delay", "period", "time", "duration"]
        }
        
        expanded = query
        for original, syns in synonyms.items():
            if original in expanded.lower():
                for syn in syns:
                    if syn not in expanded.lower():
                        expanded += f" {syn}"
        
        return expanded
    
    def get_enhanced_suggestions(self, query: str) -> List[str]:
        """Generate enhanced query suggestions for ANY question"""
        suggestions = []
        
        # Add variations with different terms
        base_query = query.lower()
        
        # Enhanced synonym expansion
        synonyms = {
            "cover": ["coverage", "include", "provide", "offer", "benefit"],
            "waiting period": ["wait", "time", "duration", "period", "delay"],
            "exclude": ["exclusion", "not covered", "restriction", "limitation"],
            "claim": ["claim process", "claim procedure", "claim submission"],
            "policy": ["insurance", "coverage", "plan", "terms"],
            "surgery": ["operation", "procedure", "treatment"],
            "hospital": ["medical center", "clinic", "healthcare facility"],
            "benefit": ["coverage", "advantage", "feature", "provision"],
            "premium": ["payment", "amount", "cost", "fee"],
            "coverage": ["benefit", "include", "cover", "provide"],
            "exclusion": ["not covered", "restriction", "limitation"],
            "waiting": ["wait", "delay", "period", "time"]
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
            query.replace("where", "in what location"),
            query.replace("why", "what is the reason"),
            query.replace("which", "what"),
            query.replace("is there", "does this include"),
            query.replace("does this", "is this"),
            query.replace("will this", "does this"),
            query.replace("can this", "does this")
        ]
        suggestions.extend(question_variations)
        
        # Add rephrasing variations
        rephrasing = [
            f"What does the policy say about {query.split()[-1]}?",
            f"How does the policy handle {query.split()[-1]}?",
            f"What are the details regarding {query.split()[-1]}?",
            f"Can you explain {query.split()[-1]} in the policy?",
            f"What is the policy's stance on {query.split()[-1]}?"
        ]
        suggestions.extend(rephrasing)
        
        return list(set(suggestions))[:10]  # Limit to 10 unique suggestions