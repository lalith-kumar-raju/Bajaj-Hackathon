from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from openai import AzureOpenAI
import numpy as np
from typing import List, Dict, Any, Optional
from models import DocumentChunk
from config import Config
import logging
import uuid
import hashlib
import time

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.config = Config()
        self.client = None
        self.embedding_client = None
        self.embedding_cache = {}  # Cache for embeddings
        self.response_cache = {}   # Cache for search responses
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Qdrant client and OpenAI embedding client"""
        try:
            # Check if Azure AI Studio credentials are available
            if not self.config.AZURE_AI_STUDIO_API_KEY:
                raise Exception("Azure AI Studio API key not configured. Please set AZURE_AI_STUDIO_API_KEY in your .env file.")
            if not self.config.AZURE_AI_STUDIO_ENDPOINT:
                raise Exception("Azure AI Studio endpoint not configured. Please set AZURE_AI_STUDIO_ENDPOINT in your .env file.")
            
            # Initialize OpenAI client for embeddings
            self.embedding_client = AzureOpenAI(
                api_key=self.config.AZURE_AI_STUDIO_API_KEY,
                api_version=self.config.AZURE_AI_STUDIO_API_VERSION,
                azure_endpoint=self.config.AZURE_AI_STUDIO_ENDPOINT
            )
            
            # Test OpenAI connection
            logger.info("🧪 Testing Azure AI Studio connection...")
            test_response = self.embedding_client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=["test"]
            )
            logger.info("✅ Azure AI Studio connection successful")
            
            # Initialize Qdrant client
            logger.info("🧪 Testing Qdrant connection...")
            if self.config.QDRANT_API_KEY:
                self.client = QdrantClient(
                    host=self.config.QDRANT_HOST,
                    port=self.config.QDRANT_PORT,
                    api_key=self.config.QDRANT_API_KEY,
                    timeout=60.0  # Reduced timeout for faster responses
                )
            else:
                # Local Qdrant without API key
                self.client = QdrantClient(
                    host=self.config.QDRANT_HOST,
                    port=self.config.QDRANT_PORT,
                    timeout=60.0  # Reduced timeout for faster responses
                )
            
            # Test Qdrant connection
            collections = self.client.get_collections()
            logger.info("✅ Qdrant connection successful")
            
            # Initialize collection
            self._initialize_collection()
            
            logger.info("✅ Qdrant and OpenAI clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing clients: {e}")
            raise Exception(f"Failed to initialize vector store clients: {str(e)}")
    
    def _initialize_collection(self):
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.config.QDRANT_COLLECTION_NAME not in collection_names:
                logger.info(f"Creating new Qdrant collection: {self.config.QDRANT_COLLECTION_NAME}")
                
                # Create collection with proper vector configuration
                self.client.create_collection(
                    collection_name=self.config.QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.config.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Successfully created Qdrant collection: {self.config.QDRANT_COLLECTION_NAME}")
            else:
                logger.info(f"✅ Using existing Qdrant collection: {self.config.QDRANT_COLLECTION_NAME}")
            
            # Test the connection
            collection_info = self.client.get_collection(self.config.QDRANT_COLLECTION_NAME)
            logger.info(f"✅ Qdrant connection verified. Collection info: {collection_info.points_count} vectors")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Qdrant collection: {e}")
            raise Exception(f"Failed to initialize Qdrant collection: {str(e)}")
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.embedding_cache.get(text_hash)
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.embedding_cache[text_hash] = embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI text-embedding-3-large with caching"""
        try:
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache first
            for i, text in enumerate(texts):
                cached_embedding = self._get_cached_embedding(text)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    embeddings.append(None)  # Placeholder
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                logger.info(f"📊 Generating embeddings for {len(uncached_texts)} uncached texts...")
                
                # Process texts in larger batches for efficiency
                batch_size = self.config.EMBEDDING_BATCH_SIZE
                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i:i + batch_size]
                    
                    response = self.embedding_client.embeddings.create(
                        model=self.config.EMBEDDING_MODEL,
                        input=batch
                    )
                    
                    batch_embeddings = [embedding.embedding for embedding in response.data]
                    
                    # Cache and store embeddings
                    for j, (text, embedding) in enumerate(zip(batch, batch_embeddings)):
                        global_index = i + j
                        original_index = uncached_indices[global_index]
                        embeddings[original_index] = embedding
                        self._cache_embedding(text, embedding)
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings using OpenAI (with caching)")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks in Qdrant vector database with optimized batching"""
        try:
            logger.info(f"🚀 Starting to store {len(chunks)} chunks in Qdrant...")
            
            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            logger.info(f"📊 Generating embeddings for {len(texts)} texts...")
            embeddings = self.generate_embeddings(texts)
            logger.info(f"✅ Generated {len(embeddings)} embeddings successfully")
            
            # Prepare points for Qdrant
            points = []
            for i, chunk in enumerate(chunks):
                # Clean metadata for Qdrant
                metadata = {
                    "content": chunk.content,
                    "document_id": chunk.metadata.get("document_id", ""),
                    "chunk_index": str(chunk.metadata.get("chunk_index", 0)),
                    "document_url": chunk.metadata.get("document_url", ""),
                    "word_count": chunk.metadata.get("word_count", 0),
                    "original_chunk_id": chunk.chunk_id  # Store original chunk ID in metadata
                }
                
                # Convert sections to string if it's a dict
                if isinstance(chunk.metadata.get("sections"), dict):
                    metadata["sections"] = str(chunk.metadata.get("sections", {}))
                
                point = PointStruct(
                    id=str(uuid.uuid4()), # Convert UUID to string for Qdrant
                    vector=embeddings[i],
                    payload=metadata
                )
                points.append(point)
            
            # Upsert points in optimized batches
            batch_size = self.config.QDRANT_BATCH_SIZE
            total_batches = (len(points) + batch_size - 1) // batch_size
            logger.info(f"📦 Upserting {len(points)} points in {total_batches} batches...")
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.info(f"📤 Processing batch {batch_num}/{total_batches} ({len(batch)} points)...")
                
                # Add retry logic for timeouts
                max_retries = 3  # Reduced retries for faster processing
                for retry in range(max_retries):
                    try:
                        self.client.upsert(
                            collection_name=self.config.QDRANT_COLLECTION_NAME,
                            points=batch,
                            wait=True  # Wait for operation to complete
                        )
                        logger.info(f"✅ Batch {batch_num} stored successfully")
                        break  # Success, exit retry loop
                    except Exception as e:
                        if ("timeout" in str(e).lower() or "timed out" in str(e).lower()) and retry < max_retries - 1:
                            logger.warning(f"⚠️  Batch {batch_num} timeout, retrying ({retry + 1}/{max_retries})...")
                            import time
                            time.sleep(5)  # Reduced wait time
                        else:
                            logger.error(f"❌ Batch {batch_num} failed after {retry + 1} attempts: {e}")
                            raise e
            
            logger.info(f"✅ Successfully stored {len(chunks)} chunks in Qdrant vector database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing chunks in Qdrant: {e}")
            raise Exception(f"Failed to store chunks in Qdrant: {str(e)}")
    
    def _get_cached_response(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search response if available"""
        if not self.config.ENABLE_RESPONSE_CACHE:
            return None
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cached_data = self.response_cache.get(query_hash)
        
        if cached_data and time.time() - cached_data['timestamp'] < self.config.RESPONSE_CACHE_TTL:
            return cached_data['response']
        
        return None
    
    def _cache_response(self, query: str, response: List[Dict[str, Any]]):
        """Cache search response"""
        if not self.config.ENABLE_RESPONSE_CACHE:
            return
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.response_cache[query_hash] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def search_similar(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Enhanced search for similar chunks using multiple strategies"""
        try:
            # Check cache first
            cached_response = self._get_cached_response(query)
            if cached_response:
                logger.info(f"✅ Using cached response for query: '{query[:50]}...'")
                return cached_response
            
            logger.info(f"🔍 Searching Qdrant for query: '{query[:50]}...'")
            
            # Strategy 1: Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            logger.info(f"📊 Generated query embedding successfully")
            
            # Strategy 2: Multiple search approaches
            search_results = []
            
            # Primary search with original query
            top_k = top_k or self.config.TOP_K_RESULTS
            logger.info(f"🔎 Querying Qdrant with top_k={top_k}...")
            
            primary_results = self.client.search(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            search_results.extend(primary_results)
            
            # Strategy 3: Enhanced query variations for better retrieval
            query_variations = self._generate_query_variations(query) if self.config.ENABLE_QUERY_VARIATIONS else []
            for variation in query_variations[:self.config.MAX_QUERY_VARIATIONS]:  # Limit to configured variations
                try:
                    variation_embedding = self.generate_embeddings([variation])[0]
                    variation_results = self.client.search(
                        collection_name=self.config.QDRANT_COLLECTION_NAME,
                        query_vector=variation_embedding,
                        limit=top_k // 2,  # Half the results for variations
                        with_payload=True
                    )
                    search_results.extend(variation_results)
                except Exception as e:
                    logger.warning(f"⚠️  Variation search failed: {e}")
            
            # Strategy 4: Process and deduplicate results
            similar_chunks = self._process_search_results(search_results)
            
            # Cache the response
            self._cache_response(query, similar_chunks)
            
            logger.info(f"✅ Found {len(similar_chunks)} similar chunks for query using enhanced search (threshold: {self.config.SIMILARITY_THRESHOLD})")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"❌ Error searching similar chunks in Qdrant: {e}")
            raise Exception(f"Failed to search chunks in Qdrant: {str(e)}")
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better retrieval"""
        variations = []
        query_lower = query.lower()
        
        # Synonym-based variations
        synonyms = {
            "cover": ["coverage", "include", "provide", "benefit"],
            "waiting period": ["wait", "time", "duration", "delay"],
            "exclude": ["exclusion", "not covered", "restriction"],
            "claim": ["claim process", "claim procedure", "submit"],
            "policy": ["insurance", "coverage", "plan"],
            "surgery": ["operation", "procedure", "treatment"],
            "hospital": ["medical center", "clinic", "healthcare"],
            "benefit": ["coverage", "advantage", "feature"],
            "premium": ["payment", "amount", "cost"],
            "coverage": ["benefit", "include", "cover"],
            "exclusion": ["not covered", "restriction", "limitation"],
            "waiting": ["wait", "delay", "period"]
        }
        
        # Generate variations with synonyms
        for original, syns in synonyms.items():
            if original in query_lower:
                for syn in syns:
                    variation = query.replace(original, syn)
                    if variation != query:
                        variations.append(variation)
        
        # Question pattern variations
        question_patterns = [
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
        variations.extend(question_patterns)
        
        # Add policy-specific variations
        policy_variations = [
            f"policy {query}",
            f"insurance {query}",
            f"coverage {query}",
            f"benefit {query}",
            f"claim {query}",
            f"exclusion {query}",
            f"waiting period {query}",
            f"premium {query}"
        ]
        variations.extend(policy_variations)
        
        return list(set(variations))  # Remove duplicates
    
    def _process_search_results(self, search_results: List) -> List[Dict[str, Any]]:
        """Process and deduplicate search results with enhanced scoring"""
        processed_chunks = {}
        
        for result in search_results:
            content = result.payload.get("content", "")
            score = result.score
            chunk_id = result.payload.get("original_chunk_id", str(result.id))
            
            # Skip if already processed with higher score
            if chunk_id in processed_chunks and processed_chunks[chunk_id]["score"] >= score:
                continue
            
            # Enhanced relevance scoring
            enhanced_score = self._calculate_enhanced_relevance_score(content, score)
            
            if enhanced_score >= self.config.SIMILARITY_THRESHOLD:
                processed_chunks[chunk_id] = {
                    "content": content,
                    "score": enhanced_score,
                    "metadata": result.payload,
                    "chunk_id": chunk_id
                }
        
        # Sort by enhanced score
        similar_chunks = list(processed_chunks.values())
        similar_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        return similar_chunks
    
    def _calculate_enhanced_relevance_score(self, content: str, base_score: float) -> float:
        """Calculate enhanced relevance score based on multiple factors"""
        enhanced_score = base_score
        
        # Factor 1: Content length (longer content might be more relevant)
        content_length_factor = min(len(content) / 1000, 0.1)  # Max 0.1 boost
        enhanced_score += content_length_factor
        
        # Factor 2: Policy-specific terms boost
        policy_terms = ["policy", "coverage", "benefit", "claim", "exclusion", "waiting period", 
                       "premium", "sum insured", "deductible", "room rent", "icu", "surgery"]
        policy_term_count = sum(1 for term in policy_terms if term.lower() in content.lower())
        policy_boost = min(policy_term_count * 0.05, 0.2)  # Max 0.2 boost
        enhanced_score += policy_boost
        
        # Factor 3: Specific details boost
        detail_indicators = ["rs", "rupees", "percent", "%", "days", "months", "years", 
                           "clause", "section", "condition", "limitation"]
        detail_count = sum(1 for indicator in detail_indicators if indicator.lower() in content.lower())
        detail_boost = min(detail_count * 0.02, 0.1)  # Max 0.1 boost
        enhanced_score += detail_boost
        
        return min(enhanced_score, 1.0)  # Cap at 1.0
    
    def search_universal(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Universal search strategy for ANY question type with caching"""
        try:
            # For Qdrant, we'll use the same semantic search as it's already optimized
            return self.search_similar(query, top_k)
            
        except Exception as e:
            logger.error(f"❌ Error in universal search in Qdrant: {e}")
            raise Exception(f"Failed to perform universal search in Qdrant: {str(e)}")
    
    def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            # Delete points with specific document_id
            self.client.delete(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            
            logger.info(f"✅ Deleted chunks for document {document_id} from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting document chunks from Qdrant: {e}")
            raise Exception(f"Failed to delete document chunks from Qdrant: {str(e)}")
    
    def clear_collection(self):
        """Clear the collection and recreate it"""
        try:
            logger.info(f"🗑️  Clearing Qdrant collection: {self.config.QDRANT_COLLECTION_NAME}")
            
            # Delete existing collection
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.config.QDRANT_COLLECTION_NAME in collection_names:
                self.client.delete_collection(self.config.QDRANT_COLLECTION_NAME)
                logger.info(f"✅ Deleted existing collection: {self.config.QDRANT_COLLECTION_NAME}")
            
            # Recreate collection
            self._initialize_collection()
            logger.info(f"✅ Recreated collection: {self.config.QDRANT_COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            raise Exception(f"Failed to clear collection: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            collection_info = self.client.get_collection(self.config.QDRANT_COLLECTION_NAME)
            return {
                "total_vectors": collection_info.points_count,
                "collection_name": self.config.QDRANT_COLLECTION_NAME,
                "dimension": collection_info.config.params.vectors.size,
                "status": "connected"
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection stats from Qdrant: {e}")
            raise Exception(f"Failed to get collection stats from Qdrant: {str(e)}")