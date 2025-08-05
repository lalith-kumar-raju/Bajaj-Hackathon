import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Azure AI Studio (AI Foundry) Configuration
    AZURE_AI_STUDIO_ENDPOINT = os.getenv("AZURE_AI_STUDIO_ENDPOINT")
    AZURE_AI_STUDIO_API_KEY = os.getenv("AZURE_AI_STUDIO_API_KEY")
    AZURE_AI_STUDIO_API_VERSION = os.getenv("AZURE_AI_STUDIO_API_VERSION", "2024-02-15-preview")
    
    # Model Names for Azure AI Studio
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Qdrant Vector Database Configuration
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    # Handle both local and cloud URLs
    if qdrant_host.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        parsed_url = urlparse(qdrant_host)
        QDRANT_HOST = parsed_url.netloc
    else:
        QDRANT_HOST = qdrant_host
    
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hackrx-documents")
    
    # HackRx API Key
    HACKRX_API_KEY = os.getenv("HACKRX_API_KEY")
    
    # Model Settings - Optimized for 100-Page PDFs
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))  # Balanced for accuracy and speed
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.05"))  # Lower for better accuracy
    
    # Enhanced Document Processing - Optimized for 100-Page PDFs
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))  # Balanced for accuracy and speed
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "250"))  # Good overlap for accuracy
    MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "800"))  # Handle 100-page PDFs
    
    # Enhanced Search Settings - Optimized for 100-Page PDFs
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))  # Good balance for accuracy
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))  # Balanced threshold
    
    # API Settings - Optimized for 100-Page PDFs
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "20"))  # Balanced timeout
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "15"))  # Balanced concurrency
    
    # Cache Settings - Enhanced for SPEED
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "7200"))  # Increased cache time
    
    # Vector Database Settings
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
    
    # Performance Optimization Settings - Optimized for 100-Page PDFs
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "40"))  # Balanced batch size
    QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "20"))  # Balanced batch size
    ENABLE_RESPONSE_CACHE = os.getenv("ENABLE_RESPONSE_CACHE", "True").lower() == "true"
    RESPONSE_CACHE_TTL = int(os.getenv("RESPONSE_CACHE_TTL", "3600"))
    
    # Enhanced Search Settings
    ENABLE_QUERY_VARIATIONS = os.getenv("ENABLE_QUERY_VARIATIONS", "True").lower() == "true"
    MAX_QUERY_VARIATIONS = int(os.getenv("MAX_QUERY_VARIATIONS", "3"))
    ENABLE_ENHANCED_SCORING = os.getenv("ENABLE_ENHANCED_SCORING", "True").lower() == "true"
    
    # Advanced LLM Settings
    ENABLE_FALLBACK_PROMPTS = os.getenv("ENABLE_FALLBACK_PROMPTS", "True").lower() == "true"
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    ENABLE_ANSWER_VALIDATION = os.getenv("ENABLE_ANSWER_VALIDATION", "True").lower() == "true"
    
    # Robustness Settings
    ENABLE_MULTIPLE_SEARCH_STRATEGIES = os.getenv("ENABLE_MULTIPLE_SEARCH_STRATEGIES", "True").lower() == "true"
    ENABLE_SYNONYM_EXPANSION = os.getenv("ENABLE_SYNONYM_EXPANSION", "True").lower() == "true"
    ENABLE_ENTITY_BOOSTING = os.getenv("ENABLE_ENTITY_BOOSTING", "True").lower() == "true"
    
    # Quality Assurance Settings
    MIN_ANSWER_LENGTH = int(os.getenv("MIN_ANSWER_LENGTH", "50"))
    MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH", "1000"))
    ENABLE_ANSWER_QUALITY_CHECK = os.getenv("ENABLE_ANSWER_QUALITY_CHECK", "True").lower() == "true"
    
    # Accuracy vs Speed Balance Settings - Optimized for 100-Page PDFs
    ACCURACY_PRIORITY = os.getenv("ACCURACY_PRIORITY", "True").lower() == "true"  # Set to True for accuracy
    MAX_QUERY_VARIATIONS = int(os.getenv("MAX_QUERY_VARIATIONS", "2")) if os.getenv("ACCURACY_PRIORITY", "True").lower() == "true" else 1
    ENABLE_ENHANCED_PROMPTS = os.getenv("ENABLE_ENHANCED_PROMPTS", "True").lower() == "true"
    ENABLE_QUERY_VARIATIONS = os.getenv("ENABLE_QUERY_VARIATIONS", "True").lower() == "true" and os.getenv("ACCURACY_PRIORITY", "True").lower() == "true"