#!/usr/bin/env python3
"""
Test script to verify the setup and basic functionality.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import DocumentProcessor: {e}")
        return False
    
    try:
        from vector_store import VectorStore
        print("✅ VectorStore imported successfully")
    except Exception as e:
        print(f"❌ Failed to import VectorStore: {e}")
        return False
    
    try:
        from main import app
        print("✅ FastAPI app imported successfully")
    except Exception as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        return False
    
    return True

async def test_environment():
    """Test if environment variables are set."""
    print("\n🔍 Testing environment variables...")
    
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "AUTH_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables.")
        return False
    
    return True

async def test_document_processor():
    """Test document processor functionality."""
    print("\n🔍 Testing document processor...")
    
    try:
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized successfully")
        
        # Test chunking functionality
        test_text = "This is a test document. It contains multiple sentences. We will test chunking."
        chunks = processor._create_chunks(test_text, "test_url")
        
        if chunks:
            print(f"✅ Text chunking works: {len(chunks)} chunks created")
        else:
            print("❌ Text chunking failed")
            return False
            
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False
    
    return True

async def test_vector_store():
    """Test vector store functionality."""
    print("\n🔍 Testing vector store...")
    
    try:
        from vector_store import VectorStore
        
        # Test with Pinecone disabled to avoid API calls
        vector_store = VectorStore(use_pinecone=False)
        print("✅ VectorStore initialized successfully (Pinecone disabled)")
        
        # Test embedding generation (this will fail without OpenAI API key)
        try:
            embeddings = await vector_store._generate_embeddings(["test text"])
            print("✅ Embedding generation works")
        except Exception as e:
            print(f"⚠️  Embedding generation failed (expected without API key): {e}")
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests."""
    print("🧪 Running Bajajhackx6 setup tests...\n")
    
    tests = [
        test_imports(),
        test_environment(),
        test_document_processor(),
        test_vector_store()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print("="*50)
    
    test_names = ["Imports", "Environment", "Document Processor", "Vector Store"]
    passed = 0
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        if isinstance(result, Exception):
            print(f"❌ {name}: Failed with exception - {result}")
        elif result:
            print(f"✅ {name}: Passed")
            passed += 1
        else:
            print(f"❌ {name}: Failed")
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your setup is ready.")
        print("\nTo start the application, run:")
        print("   python run.py")
        print("   or")
        print("   uvicorn main:app --reload")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the application.")

if __name__ == "__main__":
    asyncio.run(main()) 