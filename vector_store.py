from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from openai import AzureOpenAI
import numpy as np
from typing import List, Dict, Any, Optional
from models import DocumentChunk
from config import Config
import logging
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.config = Config()
        self.client = None
        self.embedding_client = None
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
                    timeout=120.0  # Increased to 120 seconds for large documents
                )
            else:
                # Local Qdrant without API key
                self.client = QdrantClient(
                    host=self.config.QDRANT_HOST,
                    port=self.config.QDRANT_PORT,
                    timeout=120.0  # Increased to 120 seconds for large documents
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
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI text-embedding-3-large"""
        try:
            embeddings = []
            
            # Process texts in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.embedding_client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=batch
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings using OpenAI")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks in Qdrant vector database"""
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
            
            # Upsert points in batches
            batch_size = 5  # Reduced for large documents to avoid timeouts
            total_batches = (len(points) + batch_size - 1) // batch_size
            logger.info(f"📦 Upserting {len(points)} points in {total_batches} batches...")
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.info(f"📤 Processing batch {batch_num}/{total_batches} ({len(batch)} points)...")
                
                # Add retry logic for timeouts
                max_retries = 5  # Increased retries for hackathon
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
                            time.sleep(10)  # Wait longer before retry for large docs
                        else:
                            logger.error(f"❌ Batch {batch_num} failed after {retry + 1} attempts: {e}")
                            raise e
            
            logger.info(f"✅ Successfully stored {len(chunks)} chunks in Qdrant vector database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing chunks in Qdrant: {e}")
            raise Exception(f"Failed to store chunks in Qdrant: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using semantic similarity"""
        try:
            logger.info(f"🔍 Searching Qdrant for query: '{query[:50]}...'")
            
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            logger.info(f"📊 Generated query embedding successfully")
            
            # Search in Qdrant
            top_k = top_k or self.config.TOP_K_RESULTS
            logger.info(f"🔎 Querying Qdrant with top_k={top_k}...")
            
            results = self.client.search(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            # Process results
            similar_chunks = []
            for result in results:
                if result.score >= self.config.SIMILARITY_THRESHOLD:
                    similar_chunks.append({
                        "content": result.payload.get("content", ""),
                        "score": result.score,
                        "metadata": result.payload,
                        "chunk_id": result.payload.get("original_chunk_id", str(result.id))  # Use original chunk ID from metadata
                    })
            
            logger.info(f"✅ Found {len(similar_chunks)} similar chunks for query using Qdrant (threshold: {self.config.SIMILARITY_THRESHOLD})")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"❌ Error searching similar chunks in Qdrant: {e}")
            raise Exception(f"Failed to search chunks in Qdrant: {str(e)}")
    
    def search_universal(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Universal search strategy for ANY question type"""
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