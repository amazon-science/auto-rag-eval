# Retrieval Systems

## Provided Models

This folder contains several classes of retrieval models, to be evaluated in combination with LLMs. Most notably, we provide the code for:

* **Sparse Methods**:
  * **BM25**: Classical BM25 retriver, implementation from [this repo](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py)
* **Dense Methods**:
  * **SiameseContextProvider**: Siamese mode, using `vblagoje/dpr-question_encoder-single-lfqa-base`, from Hugging Face.
  * **EmbeddingContextProvider**: Classical embedding model, using `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`, from Hugging Face.
  * Moreover, we provide a general index implementation in `docs_faiss_index.py`
* **Hybrid Methods**:
  * **DPRContextGenerator**: Cross-Encode, to aggregate base retrieval models, using `cross-encoder/ms-marco-MiniLM-L-6-v2`,  from Hugging Face.

One can use the file `test_retrieval_models.py` to test the implementation of models.

## Custom Models

To implement your own model, follow the convention from the abstract class `ContextProvider` in `context_utils.py`

## Model Usage

To leverage different models during inference and potential ensemble several models, one just need to follow the convention:

```[python]
context_generator_dict = {
    'DPR': DPRContextGenerator(context_sources={
        'SIAMESE' : SiameseContextProvider(index_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/RetrievalIndex/siamese_emb",
                                            data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main",
                                            regenerate_index=False),
        'BM25': BM25ContextProvider(data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main")
    }),
    'BM25' : BM25ContextProvider(data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main"),
    'SIAMESE' : SiameseContextProvider(index_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/RetrievalIndex/siamese_emb",
                                        data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main",
                                        regenerate_index=True),
    'MultiQA' : EmbeddingContextProvider(index_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/RetrievalIndex/multi_qa_emb",
                                            data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main",
                                            regenerate_index=True),
    'DPR:MultiQA:BM25': DPRContextGenerator(context_sources={
        'MultiQA' : EmbeddingContextProvider(index_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/RetrievalIndex/multi_qa_emb",
                                                data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main",
                                                regenerate_index=False),
        'BM25': BM25ContextProvider(data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main")
    }),
} 
```
