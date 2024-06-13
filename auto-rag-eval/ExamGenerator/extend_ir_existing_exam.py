import json
from os.path import abspath, dirname

from RetrievalSystems.bm25 import BM25ContextProvider
from RetrievalSystems.dpr_context_aggregator import DPRContextGenerator
from RetrievalSystems.embedding_retriever import EmbeddingContextProvider
from RetrievalSystems.siamese_retriever import SiameseContextProvider
from tqdm import tqdm

ROOTPATH = dirname(dirname(abspath(__file__)))

if __name__ == '__main__':

    for exam_setting in tqdm([
        {'task_domain': 'Arxiv',
         'exam_folder': 'small_llamav2_2023091905'},
        {'task_domain': 'Arxiv',
         'exam_folder': 'small_openllama_2023091905'},
    ]):

        context_generator_dict = {
            'DPR': DPRContextGenerator(context_sources={
                'SIAMESE' : SiameseContextProvider(index_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/RetrievalIndex/siamese_emb",
                                                   data_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/KnowledgeCorpus/main",
                                                   regenerate_index=False),
                'BM25': BM25ContextProvider(data_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/KnowledgeCorpus/main")
            }),
            'BM25' : BM25ContextProvider(data_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/KnowledgeCorpus/main"),
            'SIAMESE' : SiameseContextProvider(index_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/RetrievalIndex/siamese_emb",
                                               data_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/KnowledgeCorpus/main",
                                               regenerate_index=True),
            'MultiQA' : EmbeddingContextProvider(index_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/RetrievalIndex/multi_qa_emb",
                                                 data_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/KnowledgeCorpus/main",
                                                 regenerate_index=True),
            'DPR:MultiQA:BM25': DPRContextGenerator(context_sources={
                'MultiQA' : EmbeddingContextProvider(index_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/RetrievalIndex/multi_qa_emb",
                                                    data_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/KnowledgeCorpus/main",
                                                    regenerate_index=False),
                'BM25': BM25ContextProvider(data_folder=f"{ROOTPATH}/Data/{exam_setting['task_domain']}/KnowledgeCorpus/main")
            }),
        }

        with open(f"{ROOTPATH}/Data/{exam_setting['task_domain']}/ExamData/{exam_setting['exam_folder']}/exam.json", "r") as outfile:
            docs_exam = json.load(outfile)

        for question in docs_exam:

            question['retrieved_context'] = {**question['retrieved_context'],
                                             **{retriever : [elem.text for elem in context_generator.get_context_from_query(
                                                question['question'])] for retriever, context_generator in context_generator_dict.items()}}

        with open(f"{ROOTPATH}/Data/{exam_setting['task_domain']}/ExamData/{exam_setting['exam_folder']}/updated_ir_exam.json", "w") as outfile:
            outfile.write(json.dumps(docs_exam))
