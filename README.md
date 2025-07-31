# Bajajhackx6 - Policy RAG System

A FastAPI-based Retrieval Augmented Generation (RAG) system for processing insurance policy documents and answering questions using AI.

## Features

- **Document Processing**: Supports PDF and DOCX files
- **Vector Storage**: Uses Pinecone for efficient similarity search
- **AI-Powered Q&A**: Leverages OpenAI GPT models for intelligent responses
- **RESTful API**: Clean FastAPI endpoints with authentication
- **Async Processing**: Efficient handling of multiple documents and questions

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west-2

# Authentication Token (for API access)
AUTH_TOKEN=your_auth_token_here
```

### 3. Run the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Usage

### Authentication

All API endpoints require Bearer token authentication. Include the token in the Authorization header:

```
Authorization: Bearer your_auth_token_here
```

### Main Endpoint

**POST** `/api/v1/hackrx/run`

Processes documents and answers questions using RAG.

**Request Body:**
```json
{
  "documents": [
    "https://example.com/policy1.pdf",
    "https://example.com/policy2.docx"
  ],
  "questions": [
    "What is the coverage limit for property damage?",
    "What are the exclusions in this policy?"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "question": "What is the coverage limit for property damage?",
      "answer": "Based on the policy document, the coverage limit for property damage is $500,000...",
      "source_documents": [
        {
          "text": "Property damage coverage is limited to $500,000 per occurrence...",
          "document_id": "uuid-here"
        }
      ]
    }
  ]
}
```

## Project Structure

- `main.py` - FastAPI application with endpoints
- `document_processor.py` - Handles document downloading, parsing, and chunking
- `vector_store.py` - Manages vector embeddings and Pinecone operations
- `requirements.txt` - Python dependencies
- `env_example.txt` - Example environment variables

## Architecture

1. **Document Processing**: Downloads documents from URLs, extracts text, and creates overlapping chunks
2. **Vector Storage**: Generates embeddings for text chunks and stores them in Pinecone
3. **Query Processing**: Converts questions to embeddings and finds similar document chunks
4. **Answer Generation**: Uses retrieved context with OpenAI GPT to generate accurate answers

## Error Handling

The system includes comprehensive error handling for:
- Document download failures
- Unsupported file types
- API rate limits
- Vector store errors
- Authentication failures

## Development

To run in development mode with auto-reload:

```bash
uvicorn main:app --reload
```

Access the interactive API documentation at: `http://localhost:8000/docs`