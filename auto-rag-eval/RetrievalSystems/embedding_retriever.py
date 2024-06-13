import os.path
from typing import List

import numpy as np
from datasets import load_dataset
from RetrievalSystems.context_utils import ContextPassage, ContextProvider
from RetrievalSystems.docs_faiss_index import EmbedFaissIndex
from sentence_transformers import SentenceTransformer


class EmbeddingContextProvider(ContextProvider):
    def __init__(self,
                 index_folder: str,
                 data_folder: str,
                 regenerate_index: bool = True):
        """
        index_folder := f"{ROOTPATH}/Data/DevOps/RetrievalIndex/multi_qa_emb"
        data_folder := f"{ROOTPATH}/Data/DevOps/KnowledgeCorpus/main"
        """
        self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

        self.topk_embeddings = 20
        self.min_snippet_length = 20

        self.docs_data = load_dataset(data_folder,
                                      split="train",
                                      # field="data" Old artifact from BH Template
                                      )
        # Generate a new index each time to avoid using an incorrect one
        if regenerate_index or not os.path.isfile(f"{index_folder}/kilt_dpr_data.faiss"):
            faiss_index = EmbedFaissIndex()
            faiss_index.create_faiss(data_folder=data_folder,
                                     index_folder=index_folder)
        self.docs_data.load_faiss_index("embeddings",
                                        f"{index_folder}/kilt_dpr_data.faiss")
        self.columns = ['source', 'docs_id', 'title', 'section',
                        'text', 'start_character', 'end_character', 'date']

    def embed_questions_for_retrieval(self,
                                      question: str) -> np.array:
        return self.model.encode(question)

    def query_index(self,
                    query: str) -> List[ContextPassage]:
        question_embedding = self.embed_questions_for_retrieval([query])
        a, docs_passages = self.docs_data.get_nearest_examples(
            "embeddings", question_embedding, k=self.topk_embeddings)
        retrieved_examples = []
        r = list(zip(docs_passages[k] for k in self.columns))
        for i in range(self.topk_embeddings):
            retrieved_examples.append(ContextPassage(**{k: v for k, v in zip(
                self.columns, [r[j][0][i] for j in range(len(self.columns))])}))
        return retrieved_examples

    def get_context_from_query(self,
                               query: str) -> List[ContextPassage]:

        context_passages = [res for res in self.query_index(query=query)
                            if len(res.text.split()) > self.min_snippet_length][:int(self.topk_embeddings / 3)]

        return context_passages

    def get_id(self) -> str:

        return "MultiQAEmbContextProvider"
