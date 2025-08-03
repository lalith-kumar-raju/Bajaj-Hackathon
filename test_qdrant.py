#!/usr/bin/env python3
"""
Test script to verify Qdrant integration is working correctly.
This script tests the vector store initialization and basic operations.
"""

import os
import sys
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qdrant_connection():
    """Test Qdrant connection and basic operations"""
    try:
        logger.info("🧪 Testing Qdrant connection...")
        
        # Test configuration
        config = Config()
        logger.info(f"📋 Configuration loaded:")
        logger.info(f"   - Qdrant Host: {config.QDRANT_HOST}")
        logger.info(f"   - Qdrant Port: {config.QDRANT_PORT}")
        logger.info(f"   - Qdrant API Key: {'✅ Set' if config.QDRANT_API_KEY and config.QDRANT_API_KEY != 'your_qdrant_api_key_here' else '❌ Not set (using local)'}")
        logger.info(f"   - Qdrant Collection Name: {config.QDRANT_COLLECTION_NAME}")
        logger.info(f"   - Azure AI Studio Endpoint: {'✅ Set' if config.AZURE_AI_STUDIO_ENDPOINT and config.AZURE_AI_STUDIO_ENDPOINT != 'https://your-ai-studio-endpoint.openai.azure.com/' else '❌ Not set'}")
        logger.info(f"   - Azure AI Studio API Key: {'✅ Set' if config.AZURE_AI_STUDIO_API_KEY and config.AZURE_AI_STUDIO_API_KEY != 'your_azure_ai_studio_api_key_here' else '❌ Not set'}")
        
        # Check if credentials are properly set
        missing_credentials = []
        if not config.AZURE_AI_STUDIO_API_KEY:
            missing_credentials.append("AZURE_AI_STUDIO_API_KEY")
        if not config.AZURE_AI_STUDIO_ENDPOINT:
            missing_credentials.append("AZURE_AI_STUDIO_ENDPOINT")
        if not config.HACKRX_API_KEY:
            missing_credentials.append("HACKRX_API_KEY")
        
        if missing_credentials:
            logger.error(f"❌ Missing required credentials: {', '.join(missing_credentials)}")
            logger.error("📝 Please create a .env file with the following variables:")
            logger.error("   AZURE_AI_STUDIO_ENDPOINT=https://your-ai-studio-endpoint.openai.azure.com/")
            logger.error("   AZURE_AI_STUDIO_API_KEY=your_azure_ai_studio_api_key_here")
            logger.error("   AZURE_AI_STUDIO_API_VERSION=2024-02-15-preview")
            logger.error("   LLM_MODEL=gpt-4o")
            logger.error("   EMBEDDING_MODEL=text-embedding-3-large")
            logger.error("   QDRANT_HOST=localhost")
            logger.error("   QDRANT_PORT=6333")
            logger.error("   QDRANT_COLLECTION_NAME=hackrx-documents")
            logger.error("   HACKRX_API_KEY=your_hackrx_api_key_here")
            return False
        
        # Test vector store initialization
        from vector_store import VectorStore
        logger.info("🔧 Initializing VectorStore...")
        vector_store = VectorStore()
        
        # Test collection stats
        logger.info("📊 Getting collection stats...")
        stats = vector_store.get_index_stats()
        logger.info(f"✅ Collection stats: {stats}")
        
        # Test embedding generation
        logger.info("🧠 Testing embedding generation...")
        test_texts = ["This is a test document", "Another test document"]
        embeddings = vector_store.generate_embeddings(test_texts)
        logger.info(f"✅ Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
        
        # Test search (if there are any vectors in the collection)
        if stats.get("total_vectors", 0) > 0:
            logger.info("🔍 Testing search functionality...")
            results = vector_store.search_similar("test query", top_k=5)
            logger.info(f"✅ Search returned {len(results)} results")
        else:
            logger.info("ℹ️  No vectors in collection yet, skipping search test")
        
        logger.info("🎉 All Qdrant tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Qdrant test failed: {e}")
        logger.error("💡 Troubleshooting tips:")
        logger.error("   1. Make sure you have created a .env file with proper credentials")
        logger.error("   2. Ensure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        logger.error("   3. Verify your Azure AI Studio credentials are correct")
        logger.error("   4. Check that your models are deployed in Azure AI Studio")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Qdrant integration test...")
    
    success = test_qdrant_connection()
    
    if success:
        logger.info("✅ Qdrant integration is working correctly!")
        sys.exit(0)
    else:
        logger.error("❌ Qdrant integration test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 