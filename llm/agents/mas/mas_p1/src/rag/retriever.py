import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain_core.runnables import RunnableLambda

from src.config import embedder
from src.rag.embeddings import load_embeddings


class DocumentRetriever:

    def __init__(self, corpus_path="data/corpus.json"):
        self.embeddings, self.metadata = load_embeddings(corpus_path)
        self.nbrs = NearestNeighbors(
            n_neighbors=5,
            metric='cosine'
        ).fit(self.embeddings)

    def retrieve(self, query: str):
        query_embedding = np.array(
            embedder.embed_query(query),
            dtype=np.float32
        ).reshape(1, -1)
        distances, indices = self.nbrs.kneighbors(query_embedding)
        # print([self.metadata[idx]['text'] for idx in indices[0]][0])

        return [self.metadata[idx]['text'] for idx in indices[0]][0]


# class DocumentRetrieverWrapper:

#     def __init__(self):
#         self.retriever = DocumentRetriever("data/corpus.json")

#     def retrieve_and_format(self, query: str) -> str:
#         """Форматирует результаты ретривера в строку."""
#         docs = self.retriever.retrieve(query)
#         return docs

#     def as_runnable(self):
#         return RunnableLambda(self.retrieve_and_format)
