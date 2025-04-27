
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import faiss
from ..pdf_processor import process_pdfs
from ..embedding import EmbeddingModel
from ..vector_store import VectorStore
from ..rag_pipeline import RAGPipeline
from ..agents import tool_call  # Import the new tool_call function
from ..config import config
from pydantic import BaseModel
import logging
import fitz, io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Define a directory for storing PDFs
PDF_STORAGE_DIR = "uploaded_pdfs"
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)

# Initialize components
embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
vector_store = VectorStore(config.FAISS_INDEX_PATH)
rag_pipeline = RAGPipeline(embedding_model, vector_store)

class QueryRequest(BaseModel):
    query: str

@router.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process multiple PDFs."""
    try:
        vector_store.reset()
        logger.info("Reset FAISS index before processing new PDFs")
        
        pdf_paths = []
        pdf_names = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            file_path = os.path.join(PDF_STORAGE_DIR, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            pdf_paths.append(file_path)
            pdf_names.append(file.filename)
            logger.info(f"Saved file: {file_path}")

        chunks = process_pdfs(pdf_paths, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        logger.info(f"Extracted {len(chunks)} chunks from PDFs")
        
        if not chunks:
            logger.error("No chunks extracted from PDFs")
            raise HTTPException(status_code=400, detail="No text extracted from the provided PDFs")
        
        embeddings = embedding_model.encode([chunk[0] for chunk in chunks])
        vector_store.add_vectors(embeddings, chunks)
        
        return {
            "message": f"Successfully processed {len(files)} PDFs with {len(chunks)} chunks",
            "files": pdf_names
        }
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@router.post("/query")
async def query_pdf(request: QueryRequest):
    """Query the processed PDFs with improved handling."""
    try:
        # First attempt with original query
        answer, page_numbers, chunks = rag_pipeline.generate_answer(request.query, pdf_name=None)
        
        # If answer is insufficient, try with reformulated query
        if any(phrase in answer.lower() for phrase in ["no relevant", "insufficient", "not found"]):
            reformulated_query = f"Explain in detail about: {request.query}"
            answer, page_numbers, chunks = rag_pipeline.generate_answer(reformulated_query, pdf_name=None)
        
        # Prepare citations if available
        if chunks:
            relevant_pages = sorted(set(chunk[2] for chunk in chunks))
            citations = ", ".join(
                f'<a href="/pdf?page={page}&pdf_name={chunks[0][3]}">Page {page}</a>' 
                for page in relevant_pages
            )
            response = {
                "answer": answer,
                "citations": citations
            }
        else:
            response = {
                "answer": answer,
                "citations": "No specific citations found"
            }
        
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {
            "answer": "Error processing your query. Please try again or rephrase your question.",
            "citations": "None"
        }

@router.post("/summarize")
async def generate_summary():
    """Generate a comprehensive summary."""
    try:
        context, page_numbers, chunk_results = rag_pipeline.get_full_context(pdf_name=None, agent_name="summarize")
        if not context:
            return {
                "summary": "No content available for summarization.",
                "citations": "None"
            }
        
        # Use the new tool_call function
        summary = await tool_call("summarize", context)
        
        return {
            "summary": summary,
            "citations": "None"
        }
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        return {
            "summary": f"Error generating summary: {str(e)}",
            "citations": "None"
        }

@router.post("/abstract")
async def generate_abstract():
    """Generate an academic abstract."""
    try:
        context, page_numbers, chunk_results = rag_pipeline.get_full_context(pdf_name=None, agent_name="abstract")
        if not context:
            return {
                "abstract": "No content available for abstract generation.",
                "citations": "None"
            }
        
        # Use the new tool_call function
        abstract = await tool_call("abstract", context)
        
        return {
            "abstract": abstract,
            "citations": "None"
        }
    except Exception as e:
        logger.error(f"Error generating abstract: {str(e)}", exc_info=True)
        return {
            "abstract": f"Error generating abstract: {str(e)}",
            "citations": "None"
        }

@router.post("/key_findings")
async def generate_key_findings():
    """Generate key research findings."""
    try:
        context, page_numbers, chunk_results = rag_pipeline.get_full_context(pdf_name=None, agent_name="key_findings")
        if not context:
            return {
                "key_findings": "No content available for key findings.",
                "citations": "None"
            }
        
        # Use the new tool_call function
        findings = await tool_call("key_findings", context)
        
        return {
            "key_findings": findings,
            "citations": "None"
        }
    except Exception as e:
        logger.error(f"Error generating key findings: {str(e)}", exc_info=True)
        return {
            "key_findings": f"Error generating key findings: {str(e)}",
            "citations": "None"
        }

@router.post("/challenges")
async def generate_challenges():
    """Generate research challenges and limitations."""
    try:
        context, page_numbers, chunk_results = rag_pipeline.get_full_context(pdf_name=None, agent_name="challenges")
        if not context:
            return {
                "challenges": "No content available for challenges.",
                "citations": "None"
            }
        
        # Use the new tool_call function
        challenges = await tool_call("challenges", context)
        
        return {
            "challenges": challenges,
            "citations": "None"
        }
    except Exception as e:
        logger.error(f"Error generating challenges: {str(e)}", exc_info=True)
        return {
            "challenges": f"Error generating challenges: {str(e)}",
            "citations": "None"
        }

@router.post("/reset_index")
async def reset_index():
    """Reset the FAISS index and text storage."""
    try:
        vector_store.reset()
        return {"message": "FAISS index reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting index: {str(e)}")

@router.get("/get_pdf")
async def get_pdf(pdf_name: str, page: int = 1):
    """Serve the uploaded PDF with navigation to a specific page."""
    file_path = os.path.join(PDF_STORAGE_DIR, pdf_name)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=pdf_name,
            headers={"Content-Disposition": f"inline; filename={pdf_name}"}
        )
    raise HTTPException(status_code=404, detail=f"PDF not found at {file_path}")