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
    
    # Model Settings - Optimized for better accuracy
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "6000"))  # Increased for more detailed answers
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))  # Increased for more creative responses
    
    # Enhanced Document Processing - Optimized for policy documents
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))  # Increased for better context
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))  # Increased overlap
    MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "1500"))  # Increased limit
    
    # Enhanced Search Settings - Optimized for better retrieval
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "20"))  # Increased for more context
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))  # Increased for better quality
    
    # API Settings - Optimized for rate limiting
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))  # Increased timeout
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))  # Reduced for rate limiting
    
    # Cache Settings
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    # Vector Database Settings
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))