
import os.path
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from RetrievalSystems.context_utils import ContextPassage, ContextProvider
from RetrievalSystems.docs_faiss_index import DocFaissIndex
from transformers import AutoTokenizer, DPRQuestionEncoder


class SiameseContextProvider(ContextProvider):

    def __init__(self,
                 index_folder: str,
                 data_folder: str,
                 regenerate_index: bool = True):
        """
        index_folder := f"{ROOTPATH}/Data/DevOps/RetrievalIndex"
        data_folder := f"{ROOTPATH}/Data/DevOps/KnowledgeCorpus/main"
        """
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DPRQuestionEncoder.from_pretrained(
            "vblagoje/dpr-question_encoder-single-lfqa-base").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vblagoje/dpr-question_encoder-single-lfqa-base")
        _ = self.model.eval()

        self.topk_embeddings = 20
        self.min_snippet_length = 20

        self.docs_data = load_dataset(data_folder,
                                      split="train",
                                      # field="data" Old artifact from BH Template
                                      )
        # Generate a new index each time to avoid using an incorrect one
        if regenerate_index or not os.path.isfile(f"{index_folder}/kilt_dpr_data.faiss"):
            faiss_index = DocFaissIndex()
            faiss_index.create_faiss(data_folder=data_folder,
                                     index_folder=index_folder)
        self.docs_data.load_faiss_index("embeddings",
                                        f"{index_folder}/kilt_dpr_data.faiss")
        self.columns = ['source', 'docs_id', 'title', 'section',
                        'text', 'start_character', 'end_character', 'date']

    def embed_questions_for_retrieval(self,
                                      questions: List[str]) -> np.array:
        query = self.tokenizer(questions, max_length=128, padding=True,
                               truncation=True, return_tensors="pt")
        with torch.no_grad():
            q_reps = self.model(query["input_ids"].to(self.device),
                                query["attention_mask"].to(self.device)).pooler_output
        return q_reps.cpu().numpy()

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

        return "SiameseContextProvider"
