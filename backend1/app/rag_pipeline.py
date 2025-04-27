import requests
import re
from typing import List, Tuple
from .config import config
from .embedding import EmbeddingModel
from .vector_store import VectorStore
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def query_gemma(self, prompt: str) -> str:
        """Query Gemma3-27B API with improved error handling."""
        headers = {
            "Authorization": f"Bearer {config.GEMMA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gemma3-27b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                config.GEMMA_API_URL,
                json=payload,
                headers=headers,
                timeout=config.GEMMA_API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error querying Gemma: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_answer(self, query: str, pdf_name: str = None, k: int = 5) -> Tuple[str, List[int], List[Tuple[str, float, int, str]]]:
        """Generate precise answers with improved query handling."""
        logger.info(f"Processing query: {query}")
        
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        best_results = []
        
        # Search with all variations
        for q in query_variations:
            query_embedding = self.embedding_model.encode([q])
            results = self.vector_store.search(query_embedding, k, pdf_name=pdf_name)
            if results and (not best_results or results[0][1] < best_results[0][1]):
                best_results = results
        
        # Fallback if no results found
        if not best_results:
            best_results = self._get_fallback_chunks(pdf_name, k)
        
        # Prepare context
        context = self._prepare_context(best_results)
        prompt = self._build_answer_prompt(query, context)
        
        # Generate answer
        answer = self.query_gemma(prompt)
        cleaned_answer = self._clean_response(answer)
        
        return cleaned_answer, [r[2] for r in best_results], best_results
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate multiple formulations of the query."""
        variations = [query]
        
        # For definition queries
        if any(word in query.lower() for word in ["what is", "define", "definition"]):
            variations.extend([
                f"Explain: {query}",
                f"Provide detailed explanation of: {query}"
            ])
        
        # For technical queries
        if any(word in query.lower() for word in ["formula", "equation", "calculate", "derive"]):
            variations.extend([
                f"Mathematical formulation of: {query}",
                f"Explain the derivation of: {query}"
            ])
        
        # For comparison queries
        if any(word in query.lower() for word in ["compare", "difference", "similar"]):
            variations.append(f"Key differences and similarities: {query}")
        
        return variations
    
    def _get_fallback_chunks(self, pdf_name: str, k: int) -> List[Tuple[str, float, int, str]]:
        """Get fallback chunks when no results are found."""
        logger.warning("No relevant results found, using fallback chunks")
        return [
            (text, 0.0, page, name) 
            for text, page, name in zip(
                self.vector_store.texts[:k],
                self.vector_store.page_numbers[:k],
                self.vector_store.pdf_names[:k]
            )
            if pdf_name is None or name == pdf_name
        ]
    
    def _prepare_context(self, results: List[Tuple[str, float, int, str]]) -> str:
        """Prepare context from search results with prioritization."""
        technical = []
        general = []
        
        for text, score, page, name in results:
            if any(keyword in text.lower() for keyword in ["formula", "equation", "theorem", "proof"]):
                technical.append(f"From {name}, page {page}:\n{text}")
            else:
                general.append(f"From {name}, page {page}:\n{text}")
        
        return "\n\n".join(technical + general)
    
    def _build_answer_prompt(self, query: str, context: str) -> str:
        """Construct a well-structured prompt for answer generation."""
        return f"""Answer the following question based EXCLUSIVELY on the provided context. 

Question: {query}

Context:
{context}

Instructions:
1. Provide a clear, concise answer to the question
2. Include relevant details from the context
3. For technical content, preserve mathematical notation
4. If unsure, say "The information provided doesn't contain a clear answer to this question"
5. Do not invent information not present in the context"""
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response."""
        response = re.sub(r'\*\*([^\*]+)\*\*', r'\1', response)  # Remove bold
        response = re.sub(r'\*([^\*]+)\*', r'\1', response)  # Remove italics
        response = re.sub(r'^#+.*\n', '', response, flags=re.MULTILINE)  # Remove headers
        response = re.sub(r'\n{3,}', '\n\n', response).strip()  # Normalize newlines
        return response
    
    def get_full_context(self, pdf_name: str = None, max_chars: int = 10000, agent_name: str = None) -> Tuple[str, List[int], List[Tuple[str, float, int, str]]]:
        """Retrieve all chunks for the specified PDF."""
        all_chunks = []
        
        for i in range(len(self.vector_store.texts)):
            if pdf_name is None or self.vector_store.pdf_names[i] == pdf_name:
                all_chunks.append((
                    self.vector_store.texts[i],
                    0.0,  # Distance not relevant here
                    self.vector_store.page_numbers[i],
                    self.vector_store.pdf_names[i]
                ))
        
        if not all_chunks:
            logger.warning("No chunks found, falling back to all available chunks")
            all_chunks = [
                (text, 0.0, page, name) 
                for text, page, name in zip(
                    self.vector_store.texts,
                    self.vector_store.page_numbers,
                    self.vector_store.pdf_names
                )
            ]
        
        context = "\n".join([chunk[0] for chunk in all_chunks])
        page_numbers = [chunk[2] for chunk in all_chunks]
        
        if len(context) > max_chars:
            context = context[:max_chars]
            page_numbers = page_numbers[:max_chars//1000]  # Approximate
        
        return context, page_numbers, all_chunks