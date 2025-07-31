import os
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import uuid

class VectorStore:
    """
    Handles vector storage operations using Pinecone.
    Manages document embeddings and similarity search.
    """
    
    def __init__(self, use_pinecone: bool = True, index_name: str = "hackrx-policy-docs"):
        self.use_pinecone = use_pinecone
        self.index_name = index_name
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if use_pinecone:
            self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self._initialize_pinecone_index()
    
    def _initialize_pinecone_index(self):
        """Initializes or connects to the Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = self.pinecone_client.list_indexes()
            
            if self.index_name not in [index.name for index in existing_indexes]:
                # Create new index
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-ada-002 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )
                print(f"Created new Pinecone index: {self.index_name}")
            
            # Connect to the index
            self.index = self.pinecone_client.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            print(f"Error initializing Pinecone index: {e}")
            raise
    
    async def add_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """
        Adds document chunks to the vector store.
        """
        if not self.use_pinecone:
            print("Pinecone not enabled, skipping vector storage")
            return
        
        try:
            # Generate embeddings for all chunks
            texts = [chunk["text"] for chunk in chunks]
            embeddings = await self._generate_embeddings(texts)
            
            # Prepare vectors for Pinecone
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_{i}"
                
                # Combine chunk metadata with document metadata
                combined_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk["text"][:1000],  # Limit text length for metadata
                    **chunk.get("metadata", {}),
                    **(metadata or {})
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": combined_metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"Added {len(vectors)} vectors for document {document_id}")
            
        except Exception as e:
            print(f"Error adding document chunks to vector store: {e}")
            raise
    
    async def query_vector_store(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the vector store for similar documents.
        """
        if not self.use_pinecone:
            print("Pinecone not enabled, returning empty results")
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = await self._generate_embeddings([query])
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding[0],
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "text": match.metadata.get("text", ""),
                    "document_id": match.metadata.get("document_id", ""),
                    "source_url": match.metadata.get("source_url", ""),
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using OpenAI.
        """
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                model="text-embedding-ada-002",
                input=texts
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    async def delete_document(self, document_id: str):
        """
        Deletes all vectors associated with a document.
        """
        if not self.use_pinecone:
            return
        
        try:
            # Query for all vectors with the document_id
            results = self.index.query(
                vector=[0] * 1536,  # Dummy vector
                filter={"document_id": document_id},
                top_k=10000,
                include_metadata=False
            )
            
            # Delete the vectors
            if results.matches:
                vector_ids = [match.id for match in results.matches]
                self.index.delete(ids=vector_ids)
                print(f"Deleted {len(vector_ids)} vectors for document {document_id}")
            
        except Exception as e:
            print(f"Error deleting document from vector store: {e}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the vector store index.
        """
        if not self.use_pinecone:
            return {"status": "Pinecone not enabled"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {"error": str(e)}
