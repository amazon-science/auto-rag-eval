
from os.path import abspath, dirname

from RetrievalSystems.bm25 import BM25ContextProvider, BM25Okapi
from RetrievalSystems.dpr_context_aggregator import DPRContextGenerator
from RetrievalSystems.embedding_retriever import EmbeddingContextProvider
from RetrievalSystems.siamese_retriever import SiameseContextProvider

ROOTPATH = dirname(dirname(abspath(__file__)))

if __name__ == "__main__":

    bm25_context_provider = BM25ContextProvider(
        data_folder=f"{ROOTPATH}/Data/DevOps/KnowledgeCorpus/main",
        bm25algo=BM25Okapi)

    # query = "How to connect an EC2 instance to an s3 bucket ?"
    query = "Which of the following is a valid method for verifying the Availability Zone mapping on an AWS account?"

    print(bm25_context_provider.get_context_from_query(query))

    emb_context_generator = EmbeddingContextProvider(
        index_folder=f"{ROOTPATH}/Data/DevOps/RetrievalIndex/multi_qa_emb",
        data_folder=f"{ROOTPATH}/Data/DevOps/KnowledgeCorpus/main",
        regenerate_index=True)

    print(emb_context_generator.get_context_from_query("How to terminate an EC2 instance ?"))

    # Testing the Siamese Context Generator
    # ---
    siamese_context_generator = SiameseContextProvider(
        index_folder=f"{ROOTPATH}/Data/DevOps/RetrievalIndex/siamese_emb",
        data_folder=f"{ROOTPATH}/Data/DevOps/KnowledgeCorpus/main",
        regenerate_index=False)

    print(siamese_context_generator.get_context_from_query("How to terminate an EC2 instance ?"))

    # Testing the DPR Context Generator
    # ---
    dpr_context_generator = DPRContextGenerator(
        index_folder=f"{ROOTPATH}/Data/DevOps/RetrievalIndex/siamese_emb",
        data_folder=f"{ROOTPATH}/Data/DevOps/KnowledgeCorpus/main")

    print(dpr_context_generator.get_context_from_query("How to terminate an EC2 instance ?"))
