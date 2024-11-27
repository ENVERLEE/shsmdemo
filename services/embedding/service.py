from typing import List, Dict, Any, Optional
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.exceptions import EmbeddingServiceException
from core.models import EmbeddingData
from config.settings import EMBEDDING_CONFIG

class EmbeddingService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
            self.config = config or EMBEDDING_CONFIG
            self.embeddings = LlamaCppEmbeddings(
                model_path=self.config["model_path"],
                n_ctx=self.config["chunk_size"],
                verbose=True,
                n_parts=1,
                seed=0,
                f16_kv=False,
                logits_all=False,
                vocab_only=False,
                use_mlock=False,
                n_threads=4,
                n_batch=512,
                n_gpu_layers=-1,
                device="gpu"
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"]
            )

    def create_embeddings(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[EmbeddingData]:
        try:
            chunks = self.text_splitter.split_text(text)
            vectors = self.embeddings.embed_documents(chunks)

            embedding_data = []
            for chunk, vector in zip(chunks, vectors):
                data = EmbeddingData(
                    text=chunk,
                    vector=vector,
                    metadata=metadata or {}
                )
                embedding_data.append(data)
            return embedding_data

        except Exception as e:
            raise EmbeddingServiceException(f"Error creating embeddings: {str(e)}")

    def similarity_search(
        self,
        query: str,
        embedding_data: List[EmbeddingData],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embeddings.embed_query(query)

            results = []
            for data in embedding_data:
                similarity = self._calculate_similarity(query_embedding, data.vector)
                results.append({
                    "text": data.text,
                    "similarity": similarity,
                    "metadata": data.metadata
                })

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]

        except Exception as e:
            raise EmbeddingServiceException(f"Error performing similarity search: {str(e)}")

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm1 * norm2)
        except Exception as e:
            raise EmbeddingServiceException(f"Error calculating similarity: {str(e)}")
