import argparse
import logging
import os
from os.path import abspath, dirname
from typing import Dict

import faiss
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, DPRContextEncoder

ROOTPATH = dirname(dirname(abspath(__file__)))

logger = logging.getLogger(__name__)


class FaissIndex:

    def embed_passages_for_retrieval(self, passages: Dict[str, str]) -> Dict[str, np.array]:

        pass

    def create_faiss(self,
                     data_folder: str,
                     index_folder: str) -> None:

        index_file_name = f"{index_folder}/kilt_dpr_data.faiss"
        cache_file_name = f"{index_folder}/data_kilt_embedded.arrow"

        docs_data = load_dataset(data_folder,
                                 split="train",
                                 # field="data" # To be removed for BH data template, which differs from others
                                 )

        if os.path.isfile(index_file_name):
            logger.error(f"Deleting existing Faiss index: {index_file_name}")
            os.remove(index_file_name)
        if os.path.isfile(cache_file_name):
            logger.error(f"Deleting existing Faiss index cache {cache_file_name}")
            os.remove(cache_file_name)

        # TODO: asssert set(self.docs_data_columns.features.keys()) == set(self.docs_data_columns)

        paragraphs_embeddings = docs_data.map(self.embed_passages_for_retrieval,
                                              remove_columns=self.docs_data_columns,
                                              batched=True,
                                              batch_size=512,
                                              cache_file_name=cache_file_name,
                                              desc="Creating faiss index")

        # Faiss implementation of HNSW for fast approximate nearest neighbor search
        # custom_index = faiss.IndexHNSWFlat(dims, 128, faiss.METRIC_INNER_PRODUCT)
        # custom_index = faiss.IndexFlatIP(dims)
        # custom_index = faiss.index_cpu_to_all_gpus(custom_index)

        paragraphs_embeddings.add_faiss_index(
            column="embeddings",
            custom_index=faiss.IndexFlatIP(self.dims))
        paragraphs_embeddings.save_faiss_index(
            "embeddings", index_file_name)
        logger.error("Faiss index successfully created")


class DocFaissIndex(FaissIndex):

    def __init__(self,
                 ctx_encoder_name: str = "vblagoje/dpr-ctx_encoder-single-lfqa-base"):

        self.dims = 128
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(
            ctx_encoder_name)
        self.ctx_model = DPRContextEncoder.from_pretrained(
            ctx_encoder_name).to(self.device)
        _ = self.ctx_model.eval()

        self.docs_data_columns = ['source',
                                  'docs_id',
                                  'title',
                                  'section',
                                  'text',
                                  'start_character',
                                  'end_character',
                                  'date']

    def embed_passages_for_retrieval(self,
                                     passages: Dict[str, str]):
        p = self.ctx_tokenizer(passages["text"],
                               max_length=128,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")
        with torch.no_grad():
            a_reps = self.ctx_model(p["input_ids"].to("cuda:0"),
                                    p["attention_mask"].to("cuda:0")).pooler_output

        return {"embeddings": a_reps.cpu().numpy()}


class EmbedFaissIndex(FaissIndex):

    def __init__(self,
                 model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):

        self.dims = 384
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name)

        self.docs_data_columns = ['source',
                                  'docs_id',
                                  'title',
                                  'section',
                                  'text',
                                  'start_character',
                                  'end_character',
                                  'date']

    def embed_passages_for_retrieval(self, examples):
        return {"embeddings": self.model.encode(examples['text'])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates Faiss Docs index file")

    parser.add_argument(
        "--task-domain",
        help="Task Domain, among DevOps, StackExchange...",
    )

    main_args, _ = parser.parse_known_args()

    faiss_index = DocFaissIndex()
    faiss_index.create_faiss(data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main",
                             index_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/RetrievalIndex")
