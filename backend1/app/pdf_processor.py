import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
import logging
import pytesseract
from PIL import Image
import io
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF with OCR fallback."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            
            if not page_text.strip():
                logger.info(f"Using OCR for page {page_num + 1}")
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                page_text = pytesseract.image_to_string(img)
            
            text += f"Page {page_num + 1}:\n{page_text}\n\n"
        
        doc.close()
        return text.strip()
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        raise

def process_pdfs(pdf_paths: List[str], chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, str]]:
    """Process PDFs into chunks with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            pdf_name = os.path.basename(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                if not page_text.strip():
                    logger.info(f"Using OCR for page {page_num + 1} of {pdf_name}")
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    page_text = pytesseract.image_to_string(img)
                
                if not page_text.strip():
                    continue
                
                chunks = text_splitter.split_text(page_text)
                all_chunks.extend([(chunk, page_num + 1, pdf_name) for chunk in chunks])
            
            doc.close()
            logger.info(f"Processed {pdf_name} into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            continue
    
    if not all_chunks:
        logger.error("No chunks generated from any PDFs")
        raise ValueError("No text extracted from the provided PDFs")
    
    return all_chunks
