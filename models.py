from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class QueryIntent(str, Enum):
    COVERAGE_CHECK = "coverage_check"
    WAITING_PERIOD = "waiting_period"
    EXCLUSION_CHECK = "exclusion_check"
    CLAIM_PROCESS = "claim_process"
    POLICY_DETAILS = "policy_details"
    GENERAL_QUERY = "general_query"

class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the document (PDF/DOCX)")
    questions: List[str] = Field(..., description="List of questions to process")

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to questions")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    confidence_scores: Optional[List[float]] = Field(None, description="Confidence scores for each answer")

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunk_id: str

class QueryAnalysis(BaseModel):
    intent: QueryIntent
    entities: Dict[str, Any]
    confidence: float
    processed_query: str

class DecisionResult(BaseModel):
    decision: str
    confidence: float
    supporting_clauses: List[str]
    contradicting_clauses: List[str]
    reasoning: str
    source_sections: List[str]

class ProcessingStatus(BaseModel):
    status: str
    message: str
    progress: Optional[float] = None