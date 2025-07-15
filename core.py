
import os
import fitz  # PyMuPDF
import requests
from typing import List, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult, Document
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import logging

logger = logging.getLogger(__name__)

class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter API"""

    api_key: str = Field(...)
    model: str = Field(default="meta-llama/llama-3.3-8b-instruct:free")
    base_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenRouter API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/abdul183/PDF-SUMMARIZATION---QA", 
            "X-Title": "PDF Q&A Tool" 
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.JSONDecodeError:
            logger.error("Failed to decode JSON from API response.")
            logger.error(f"Response status code: {response.status_code}")
            logger.error(f"Response text: {response.text}")
            return f"Error: The AI service returned an invalid (non-JSON) response. Status: {response.status_code}. Body: {response.text[:200]}"
        except requests.exceptions.Timeout as e:
            logger.error(f"API request timed out: {str(e)}")
            return "Error: The request to the AI service timed out. Please try again."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"API connection error: {str(e)}")
            return "Error: Could not connect to the AI service. Please check your internet connection."
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if e.response is not None:
                 return f"Error: API request failed with status {e.response.status_code}: {e.response.text}"
            return f"Error: An API request failed: {str(e)}"
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid API response format: {str(e)}")
            return "Error: The AI service returned an invalid response format after successful request."

class PDFProcessor:
    """Handles PDF processing, text extraction, and vector store creation."""

    def __init__(self, embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': 'cpu'})
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_file_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return ""

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self.text_splitter.split_text(text)

    def split_text_with_metadata(self, text: str, metadata: dict) -> List[Document]:
        """Split text into chunks and attach metadata."""
        # Create documents and then split them
        docs = self.text_splitter.create_documents([text], metadatas=[metadata])
        return self.text_splitter.split_documents(docs)

    def create_vector_store_with_metadata(self, documents: List[Document]) -> Optional[FAISS]:
        """Create FAISS vector store from documents with metadata."""
        try:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            return vectorstore
        except Exception as e:
            logger.error(f"Vector store creation error: {str(e)}")
            return None

    def create_vector_store(self, chunks: List[str]) -> Optional[FAISS]:
        """Create FAISS vector store from text chunks"""
        try:
            vectorstore = FAISS.from_texts(chunks, self.embeddings)
            return vectorstore
        except Exception as e:
            logger.error(f"Vector store creation error: {str(e)}")
            return None
