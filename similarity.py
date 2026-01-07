"""
SimilarityAgent: Handles semantic similarity search using BGE-M3 embeddings and FAISS
"""
import sys
import os
import pickle
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from models.schemas import SimilarPolicy


class SimilarityAgentError(Exception):
    """Base exception for SimilarityAgent errors"""
    pass


class EmbeddingError(SimilarityAgentError):
    """Raised when embedding generation fails"""
    pass


class IndexError(SimilarityAgentError):
    """Raised when FAISS index operations fail"""
    pass


class SimilarityAgent:
    """
    Agent responsible for semantic similarity search using local BGE-M3 model
    and FAISS vector index.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize SimilarityAgent with model and index paths.
        
        Args:
            model_path: Path to BGE-M3 model (defaults to settings.BGE_M3_PATH)
            index_path: Path to FAISS index file (defaults to settings.FAISS_INDEX_PATH)
            metadata_path: Path to metadata pickle file (defaults to settings.METADATA_PATH)
            top_k: Number of similar policies to return (defaults to settings.TOP_K_SIMILAR)
        """
        self.model_path = model_path or settings.BGE_M3_PATH
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.metadata_path = metadata_path or settings.METADATA_PATH
        self.top_k = top_k or settings.TOP_K_SIMILAR
        
        # Initialize model and index (lazy loading)
        self._model = None
        self._index = None
        self._metadata = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load BGE-M3 model"""
        if self._model is None:
            try:
                print(f"Loading BGE-M3 model from {self.model_path}...")
                self._model = SentenceTransformer(self.model_path)
                print("BGE-M3 model loaded successfully")
            except Exception as e:
                raise EmbeddingError(f"Failed to load BGE-M3 model: {str(e)}")
        return self._model
    
    @property
    def index(self) -> faiss.Index:
        """Lazy load FAISS index"""
        if self._index is None:
            try:
                if not os.path.exists(self.index_path):
                    raise IndexError(f"FAISS index not found at {self.index_path}")
                
                print(f"Loading FAISS index from {self.index_path}...")
                self._index = faiss.read_index(self.index_path)
                
                # Validate index
                if self._index.ntotal == 0:
                    raise IndexError("FAISS index is empty")
                
                print(f"FAISS index loaded successfully ({self._index.ntotal} vectors)")
            except Exception as e:
                if isinstance(e, IndexError):
                    raise
                raise IndexError(f"Failed to load FAISS index: {str(e)}")
        return self._index
    
    @property
    def metadata(self) -> Dict[int, Dict[str, Any]]:
        """Lazy load metadata mapping"""
        if self._metadata is None:
            try:
                if not os.path.exists(self.metadata_path):
                    raise IndexError(f"Metadata file not found at {self.metadata_path}")
                
                print(f"Loading metadata from {self.metadata_path}...")
                with open(self.metadata_path, 'rb') as f:
                    self._metadata = pickle.load(f)
                
                print(f"Metadata loaded successfully ({len(self._metadata)} entries)")
            except Exception as e:
                if isinstance(e, IndexError):
                    raise
                raise IndexError(f"Failed to load metadata: {str(e)}")
        return self._metadata
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for given text using BGE-M3.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        
        try:
            # Generate embedding with normalization
            embedding = self.model.encode(
                [text.strip()],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embedding.astype('float32')
            
        except MemoryError:
            raise EmbeddingError("Insufficient memory for embedding generation")
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    def find_similar(
        self,
        policy_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: float = 0.0
    ) -> List[SimilarPolicy]:
        """
        Find top-K most similar policies to the given text.
        
        Args:
            policy_text: Policy text to find similar policies for
            top_k: Number of similar policies to return (defaults to self.top_k)
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of SimilarPolicy objects, sorted by similarity score (descending)
            
        Raises:
            EmbeddingError: If embedding generation fails
            IndexError: If FAISS search fails
        """
        k = top_k or self.top_k
        
        # Generate embedding for query text
        embedding = self.generate_embedding(policy_text)
        
        # Search FAISS index
        try:
            # FAISS search returns distances and indices
            # For IndexFlatIP (inner product), higher is more similar
            distances, indices = self.index.search(embedding, k)
            
        except Exception as e:
            raise IndexError(f"FAISS search failed: {str(e)}")
        
        # Format results
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            # Convert distance to similarity score
            # For inner product with normalized vectors, distance is cosine similarity
            similarity_score = float(distance)
            
            # Skip if below threshold
            if similarity_score < similarity_threshold:
                continue
            
            # Get metadata for this policy
            idx_int = int(idx)
            if idx_int not in self.metadata:
                print(f"Warning: Index {idx_int} not found in metadata, skipping")
                continue
            
            policy_info = self.metadata[idx_int]
            
            # Create SimilarPolicy object
            similar_policy = SimilarPolicy(
                rank=rank,
                similarity_score=similarity_score,
                policy_id=policy_info.get('id', idx_int),
                country_code=policy_info.get('country_code', 'UNKNOWN'),
                policy_name=policy_info.get('policy_name', 'Unknown Policy'),
                year=policy_info.get('year'),
                instrument_type=policy_info.get('instrument_type', []),
                target_sector=policy_info.get('target_sector', [])
            )
            
            results.append(similar_policy)
        
        return results
    
    def add_to_index(
        self,
        policy_text: str,
        policy_id: int,
        policy_info: Dict[str, Any]
    ) -> int:
        """
        Add a new policy to the FAISS index and update metadata.
        
        Args:
            policy_text: Policy text to embed and add
            policy_id: Database ID of the policy
            policy_info: Metadata dictionary with keys:
                - id: int
                - country_code: str
                - policy_name: str
                - year: int (optional)
                - instrument_type: List[str] (optional)
                - target_sector: List[str] (optional)
                
        Returns:
            Index position of the added vector
            
        Raises:
            EmbeddingError: If embedding generation fails
            IndexError: If index update fails
        """
        # Generate embedding
        embedding = self.generate_embedding(policy_text)
        
        try:
            # Add to FAISS index
            self.index.add(embedding)
            
            # Get the new index position (last position)
            new_idx = self.index.ntotal - 1
            
            # Update metadata
            self.metadata[new_idx] = {
                'id': policy_id,
                'country_code': policy_info.get('country_code', 'UNKNOWN'),
                'policy_name': policy_info.get('policy_name', 'Unknown Policy'),
                'year': policy_info.get('year'),
                'instrument_type': policy_info.get('instrument_type', []),
                'target_sector': policy_info.get('target_sector', [])
            }
            
            # Persist index and metadata
            self._persist_index()
            self._persist_metadata()
            
            print(f"Added policy {policy_id} to index at position {new_idx}")
            return new_idx
            
        except Exception as e:
            raise IndexError(f"Failed to add policy to index: {str(e)}")
    
    def _persist_index(self):
        """Save FAISS index to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
        except Exception as e:
            raise IndexError(f"Failed to save FAISS index: {str(e)}")
    
    def _persist_metadata(self):
        """Save metadata to disk"""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            raise IndexError(f"Failed to save metadata: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'index_type': type(self.index).__name__,
            'metadata_entries': len(self.metadata),
            'model_path': self.model_path,
            'index_path': self.index_path
        }
