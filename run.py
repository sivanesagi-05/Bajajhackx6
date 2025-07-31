#!/usr/bin/env python3
"""
Simple script to run the FastAPI application.
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "AUTH_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file with the required variables.")
        print("See env_example.txt for reference.")
        exit(1)
    
    print("🚀 Starting Bajajhackx6 Policy RAG System...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔗 ReDoc will be available at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 