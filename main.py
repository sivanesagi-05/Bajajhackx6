import os
import asyncio
from typing import List
from uuid import uuid4

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, status, Header
from pydantic import BaseModel, HttpUrl # Ensure HttpUrl is imported
from openai import OpenAI
from pinecone import Pinecone # Import Pinecone for potential direct check, though vector_store handles it

# Import your custom modules
from document_processor import DocumentProcessor
from vector_store import VectorStore

# --- Environment Variables ---
# Ensure these are set in your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN") # The fixed auth token for the challenge
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "gcp-starter" or "us-west-2"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 Policy RAG System",
    description="API for ingesting insurance policy documents and answering questions using RAG.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Dependencies ---
# Initialize DocumentProcessor and VectorStore globally or as dependencies
# Initialize outside functions for single instantiation
document_processor = DocumentProcessor()
# Ensure use_pinecone is True as confirmed
vector_store = VectorStore(use_pinecone=True)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# --- Authentication Dependency ---
async def verify_token(authorization: str = Header(...)):
    """
    Authenticates requests using a Bearer token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or invalid. Use Bearer token."
        )
    token = authorization.split(" ")[1]
    if token != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication token."
        )
    return True

# --- Pydantic Models for Request/Response ---
class DocumentInput(BaseModel):
    documents: List[HttpUrl]
    questions: List[str]

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    source_documents: List[Dict[str, str]] # To include relevant chunks and metadata

class HackRxResponse(BaseModel):
    results: List[QuestionAnswer]

# --- Helper Function for RAG (Retrieval Augmented Generation) ---
async def generate_answer_with_rag(question: str) -> QuestionAnswer:
    """
    Retrieves relevant chunks from the vector store and uses LLM to generate an answer.
    """
    retrieved_chunks_data = await vector_store.query_vector_store(question, top_k=5)

    if not retrieved_chunks_data:
        # Fallback if no relevant documents found
        return QuestionAnswer(
            question=question,
            answer="I'm sorry, I couldn't find relevant information in the documents to answer this question.",
            source_documents=[]
        )

    # Compile context from retrieved chunks
    context_texts = [chunk["text"] for chunk in retrieved_chunks_data]
    context_str = "\n\n".join(context_texts)

    # Construct the prompt for the LLM
    prompt = f"""You are a helpful AI assistant. Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and to the point.

Context:
{context_str}

Question: {question}

Answer:"""

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-3.5-turbo", # Or "gpt-4" for better quality if available/needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500 # Adjust as needed for concise answers
        )
        answer = response.choices[0].message.content.strip()

        # Format source documents for the response
        source_docs_formatted = [
            {"text": chunk["text"], "document_id": chunk["document_id"]}
            for chunk in retrieved_chunks_data
        ]

        return QuestionAnswer(
            question=question,
            answer=answer,
            source_documents=source_docs_formatted
        )
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer with LLM: {e}")

# --- API Endpoints ---

@app.get("/")
async def root():
    """Basic endpoint to check if the server is running."""
    return {"message": "Hello, HackRx 6.0! Welcome to the Policy RAG System."}

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_hackrx_rag(
    input_data: DocumentInput,
    auth_verified: bool = Depends(verify_token) # Apply authentication
):
    """
    Ingests multiple documents and answers multiple questions using RAG.
    """
    if not auth_verified:
        # This branch should theoretically not be hit due to Depends, but as a safeguard
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed.")

    all_qa_results: List[QuestionAnswer] = []

    # 1. Document Ingestion (Process each document URL)
    print(f"Processing {len(input_data.documents)} documents...")
    for doc_url in input_data.documents:
        try:
            # Generate a unique ID for each document to use in the vector store
            document_id = str(uuid4())
            print(f"Processing document: {doc_url} with ID: {document_id}")

            # Process the document (download, parse, chunk)
            chunks = await document_processor.process_document(str(doc_url)) # Ensure URL is string
            print(f"Extracted {len(chunks)} chunks from {doc_url}")

            # Add chunks to the vector store (generates embeddings and upserts)
            if chunks:
                await vector_store.add_document_chunks(document_id, chunks, metadata={"source_url": str(doc_url)})
                print(f"Chunks from {doc_url} added to vector store.")
            else:
                print(f"No chunks extracted from {doc_url}. Skipping vector storage.")

        except Exception as e:
            print(f"Error processing document {doc_url}: {e}")
            # Continue processing other documents even if one fails
            # You might want to log this error more robustly

    # 2. Answer Questions using RAG
    print(f"Answering {len(input_data.questions)} questions...")
    for question in input_data.questions:
        try:
            qa_result = await generate_answer_with_rag(question)
            all_qa_results.append(qa_result)
        except Exception as e:
            print(f"Error answering question '{question}': {e}")
            # Append a default error response for this question
            all_qa_results.append(QuestionAnswer(
                question=question,
                answer=f"An error occurred while trying to answer this question: {e}",
                source_documents=[]
            ))

    return HackRxResponse(results=all_qa_results)