import requests
import PyPDF2
import fitz  # PyMuPDF
from docx import Document
import email
import io
import re
from typing import List, Dict, Any, Optional
from models import DocumentChunk
from config import Config
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class DocumentProcessor:
    def __init__(self):
        self.config = Config()
    
    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading document from {url}: {e}")
            raise
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        # Try PyMuPDF first (better text extraction)
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    text += page.extract_text()
            except Exception as e2:
                logger.error(f"Both PDF extractors failed: {e2}")
                raise
        
        return self.clean_text(text)
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            raise
    
    def extract_text_from_email(self, content: bytes) -> str:
        """Extract text from email content"""
        try:
            msg = email.message_from_bytes(content)
            text = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode()
            else:
                text = msg.get_payload(decode=True).decode()
            
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from email: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        # Normalize line breaks
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()
    
    def chunk_text(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Split text into semantic chunks"""
        chunks = []
        words = text.split()
        
        chunk_size = self.config.CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            chunk_id = f"{document_id}_chunk_{i//(chunk_size - overlap)}"
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    "document_id": document_id,
                    "chunk_index": i//(chunk_size - overlap),
                    "start_word": i,
                    "end_word": min(i + chunk_size, len(words)),
                    "word_count": len(chunk_words)
                },
                chunk_id=chunk_id
            )
            chunks.append(chunk)
        
        return chunks[:self.config.MAX_CHUNKS_PER_DOCUMENT]
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract policy sections based on headers"""
        sections = {}
        lines = text.split('\n')
        current_section = "General"
        current_content = []
        
        section_patterns = [
            r'^\d+\.\s*([A-Z][A-Z\s]+)',
            r'^([A-Z][A-Z\s]+):',
            r'^Section\s+\d+:\s*([A-Z][A-Z\s]+)',
            r'^Chapter\s+\d+:\s*([A-Z][A-Z\s]+)'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous section
                    if current_content:
                        sections[current_section] = ' '.join(current_content)
                    
                    current_section = match.group(1).strip()
                    current_content = []
                    is_header = True
                    break
            
            if not is_header:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = ' '.join(current_content)
        
        return sections
    
    def process_document(self, url: str) -> List[DocumentChunk]:
        """Main method to process document from URL"""
        # logger.info(f"Processing document from URL: {url}")
        
        # Download document
        content = self.download_document(url)
        
        # Determine file type and extract text
        if url.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(content)
        elif url.lower().endswith(('.docx', '.doc')):
            text = self.extract_text_from_docx(content)
        elif 'email' in url.lower() or url.lower().endswith('.eml'):
            text = self.extract_text_from_email(content)
        else:
            # Try PDF as default
            text = self.extract_text_from_pdf(content)
        
        # Generate document ID
        document_id = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Extract sections
        sections = self.extract_sections(text)
        
        # Create chunks
        chunks = self.chunk_text(text, document_id)
        
        # Add section information to metadata
        for chunk in chunks:
            chunk.metadata["sections"] = sections
            chunk.metadata["document_url"] = url
        
        logger.info(f"Processed document into {len(chunks)} chunks")
        return chunks