import faiss
import numpy as np
import json
import os
import logging
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_path: str, dimension: int = 384):
        self.index_path = index_path
        self.texts_path = f"{index_path}_texts.json"
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.page_numbers = []
        self.pdf_names = []
        
        self._load_existing_index()

    def _load_existing_index(self):
        """Load existing index if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.texts_path, 'r') as f:
                    data = json.load(f)
                    self.texts = data.get("texts", [])
                    self.page_numbers = data.get("page_numbers", [])
                    self.pdf_names = data.get("pdf_names", [])
                logger.info(f"Loaded index with {len(self.texts)} vectors")
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                self.reset()

    def add_vectors(self, vectors: np.ndarray, texts: List[Tuple[str, int, str]]):
        """Add vectors with associated metadata to the index."""
        if vectors.shape[0] == 0:
            logger.error("No vectors to add")
            return
        
        if not isinstance(texts[0], tuple) or len(texts[0]) != 3:
            logger.error("Invalid texts format")
            return
        
        self.index.add(vectors)
        new_texts, new_pages, new_pdfs = zip(*texts)
        self.texts.extend(new_texts)
        self.page_numbers.extend(new_pages)
        self.pdf_names.extend(new_pdfs)
        
        self._save_index()

    def _save_index(self):
        """Save the index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.texts_path, 'w') as f:
                json.dump({
                    "texts": self.texts,
                    "page_numbers": self.page_numbers,
                    "pdf_names": self.pdf_names
                }, f)
            logger.info(f"Saved index with {len(self.texts)} vectors")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")

    def search(self, query_vector: np.ndarray, k: int = 5, pdf_name: str = None) -> List[Tuple[str, float, int, str]]:
        """Search the index with optional PDF filtering."""
        distances, indices = self.index.search(query_vector, k*3)  # Search more to filter
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.texts):
                continue
            
            # Apply PDF filter if specified
            if pdf_name is not None and self.pdf_names[idx] != pdf_name:
                continue
                
            results.append((
                self.texts[idx],
                float(distances[0][i]),
                self.page_numbers[idx],
                self.pdf_names[idx]
            ))
        
        # Return top k results after filtering
        return sorted(results, key=lambda x: x[1])[:k]

    def reset(self):
        """Reset the index and clear all stored data."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.page_numbers = []
        self.pdf_names = []
        
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.texts_path):
            os.remove(self.texts_path)
        
        logger.info("Vector store reset complete")