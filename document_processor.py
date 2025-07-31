import asyncio
import aiohttp
import tempfile
import os
from typing import List, Dict, Any
from PyPDF2 import PdfReader
import docx
import re

class DocumentProcessor:
    """
    Handles document processing including downloading, parsing, and chunking.
    Supports PDF and DOCX files.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def process_document(self, url: str) -> List[Dict[str, Any]]:
        """
        Downloads and processes a document from a URL.
        Returns a list of text chunks with metadata.
        """
        try:
            # Download the document
            content = await self._download_document(url)
            
            # Extract text based on file type
            text = await self._extract_text(content, url)
            
            # Chunk the text
            chunks = self._create_chunks(text, url)
            
            return chunks
            
        except Exception as e:
            print(f"Error processing document {url}: {e}")
            return []
    
    async def _download_document(self, url: str) -> bytes:
        """Downloads a document from a URL."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download document: {response.status}")
                return await response.read()
    
    async def _extract_text(self, content: bytes, url: str) -> str:
        """Extracts text from document content based on file type."""
        file_extension = url.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self._extract_text_from_pdf(content)
        elif file_extension in ['docx', 'doc']:
            return self._extract_text_from_docx(content)
        else:
            # Try to decode as text
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                raise Exception(f"Unsupported file type: {file_extension}")
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extracts text from PDF content."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            reader = PdfReader(temp_file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        finally:
            os.unlink(temp_file_path)
    
    def _extract_text_from_docx(self, content: bytes) -> str:
        """Extracts text from DOCX content."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            doc = docx.Document(temp_file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        finally:
            os.unlink(temp_file_path)
    
    def _create_chunks(self, text: str, source_url: str) -> List[Dict[str, Any]]:
        """Creates overlapping chunks from text."""
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_url": source_url,
                        "chunk_index": len(chunks),
                        "start_char": start,
                        "end_char": end
                    }
                })
            
            # Move start position for next chunk
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
