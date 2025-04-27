from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def encode(self, texts: list) -> np.ndarray:
        """Encode texts into embeddings."""
        if not texts:
            logger.error("No texts provided for encoding")
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            logger.info(f"Encoded {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
