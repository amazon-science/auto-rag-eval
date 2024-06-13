import json
from os.path import abspath, dirname

from ExamAnalysis.item_response_models import (
    ExamSetting,
    HierarchicalItemResponseModel,
    ItemResponseModel,
)
from tqdm import tqdm


def get_all_students(model, task):

    root_path = f'{dirname(dirname(abspath(__file__)))}/Data/{task}/EvalResults'
    extended_students = [
        [ExamSetting(path_pattern=f'{root_path}/{task}Exam/llamav2/13b/full_sample_{task}Exam_closed_book_{model}_results_*_icl{i}.jsonl',
                     llm='llamav2:13B',
                     retrieval='closed_book',
                     icl=i,
                     name=f'Closed Book@{i} [13B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}RagExam/llamav2/13b/full_sample_{task}Exam_rag_siamese_{model}_results_*_icl{i}.jsonl',
                     llm='llamav2:13B',
                     retrieval='rag_siamese',
                     icl=i,
                     name=f'Rag Siamese@{i} [13B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}RagExam/llamav2/13b/full_sample_{task}Exam_rag_dpr_{model}_results_*_icl{i}.jsonl',
                     llm='llamav2:13B',
                     retrieval='rag_dpr',
                     icl=i,
                     name=f'Rag DPR@{i} [13B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}RagExam/llamav2/13b/full_sample_{task}Exam_rag_bm25_{model}_results_*_icl{i}.jsonl',
                     llm='llamav2:13B',
                     retrieval='rag_bm25',
                     icl=i,
                     name=f'Rag BM25@{i} [13B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/llamav2/13b/full_sample_{task}Exam_rag_multi_qa_{model}_results_*_icl{i}.jsonl',
                     llm='llamav2:13B',
                     retrieval='rag_multi_qa',
                     icl=i,
                     name=f'Rag MultiQA@{i} [13B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/llamav2/13b/full_sample_{task}Exam_rag_dpr_bm25_multi_qa_{model}_results_*_icl{i}.jsonl',
                     llm='llamav2:13B',
                     retrieval='rag_dprv2',
                     icl=i,
                     name=f'Rag DPRV2@{i} [13B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}Exam/llamav2/13b/full_sample_{task}Exam_open_book_{model}_results_*_icl{i}.jsonl',
                     llm='llamav2:13B',
                     retrieval='open_book',
                     icl=i,
                     name=f'Open Book@{i} [13B]')]
        for i in range(3)
    ]

    # Add 70B Models
    extended_students.extend([[
        ExamSetting(path_pattern=f'{root_path}/{task}Exam/llamav2/70b/full_sample_{task}Exam_closed_book_{model}_results_*_icl{i}.jsonl',
                    llm='llamav2:70B',
                    retrieval='closed_book',
                    icl=i,
                    name=f'Closed Book@{i} [70B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}RagExam/llamav2/70b/full_sample_{task}Exam_rag_siamese_{model}_results_*_icl{i}.jsonl',
                    llm='llamav2:70B',
                    retrieval='rag_siamese',
                    icl=i,
                    name=f'Rag Siamese@{i} [70B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}RagExam/llamav2/70b/full_sample_{task}Exam_rag_dpr_{model}_results_*_icl{i}.jsonl',
                    llm='llamav2:70B',
                    retrieval='rag_dpr',
                    icl=i,
                    name=f'Rag DPR@{i} [70B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}RagExam/llamav2/70b/full_sample_{task}Exam_rag_bm25_{model}_results_*_icl{i}.jsonl',
                    llm='llamav2:70B',
                    retrieval='rag_bm25',
                    icl=i,
                    name=f'Rag BM25@{i} [70B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/llamav2/70b/full_sample_{task}Exam_rag_multi_qa_{model}_results_*_icl{i}.jsonl',
                    llm='llamav2:70B',
                    retrieval='rag_multi_qa',
                    icl=i,
                    name=f'Rag MultiQA@{i} [70B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/llamav2/70b/full_sample_{task}Exam_rag_dpr_bm25_multi_qa_{model}_results_*_icl{i}.jsonl',
                    llm='llamav2:70B',
                    retrieval='rag_dprv2',
                    icl=i,
                    name=f'Rag DPRV2@{i} [70B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}Exam/llamav2/70b/full_sample_{task}Exam_open_book_{model}_results_*_icl{i}.jsonl',
                    llm='llamav2:70B',
                    retrieval='open_book',
                    icl=i,
                    name=f'Open Book@{i} [70B]')] for i in range(3)],
    )

    # Add Mistral:7B Models
    extended_students.extend([[
        ExamSetting(path_pattern=f'{root_path}/{task}Exam/mistral/7b/full_sample_{task}Exam_closed_book_{model}_results_*_icl{i}.jsonl',
                    llm='mistral:7b',
                    retrieval='closed_book',
                    icl=i,
                    name=f'Closed Book@{i} [7B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}RagExam/mistral/7b/full_sample_{task}Exam_rag_siamese_{model}_results_*_icl{i}.jsonl',
                    llm='mistral:7b',
                    retrieval='rag_siamese',
                    icl=i,
                    name=f'Rag Siamese@{i} [7B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}RagExam/mistral/7b/full_sample_{task}Exam_rag_dpr_{model}_results_*_icl{i}.jsonl',
                    llm='mistral:7b',
                    retrieval='rag_dpr',
                    icl=i,
                    name=f'Rag DPR@{i} [7B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}RagExam/mistral/7b/full_sample_{task}Exam_rag_bm25_{model}_results_*_icl{i}.jsonl',
                    llm='mistral:7b',
                    retrieval='rag_bm25',
                    icl=i,
                    name=f'Rag BM25@{i} [7B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/mistral/7b/full_sample_{task}Exam_rag_multi_qa_{model}_results_*_icl{i}.jsonl',
                    llm='mistral:7b',
                    retrieval='rag_multi_qa',
                    icl=i,
                    name=f'Rag MultiQA@{i} [7B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/mistral/7b/full_sample_{task}Exam_rag_dpr_bm25_multi_qa_{model}_results_*_icl{i}.jsonl',
                    llm='mistral:7b',
                    retrieval='rag_dprv2',
                    icl=i,
                    name=f'Rag DPRV2@{i} [7B]'),
        ExamSetting(path_pattern=f'{root_path}/{task}Exam/mistral/7b/full_sample_{task}Exam_open_book_{model}_results_*_icl{i}.jsonl',
                    llm='mistral:7b',
                    retrieval='open_book',
                    icl=i,
                    name=f'Open Book@{i} [7B]')] for i in range(3)],
    )

    return [i for elem in extended_students for i in elem]


def print_nested_dict(d, indent=0):
    """Recursively prints nested dictionaries with increasing indentation."""
    for key, value in d.items():
        print('   ' * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent + 1)
        else:
            print('   ' * (indent + 1) + (f"{value:.02f}" if type(value) != str else value))


if __name__ == '__main__':

    MODELS = ["llamav2"]
    TASKS = ['StackExchange', 'Arxiv', 'SecFilings']
    IRT_MODELS = [3]

    for task in tqdm(TASKS):

        all_stats = {}
        task_path = f"{dirname(dirname(abspath(__file__)))}/Data/{task}/EvalResults/IRT"

        for llm_model in MODELS:

            for irt_model_type in IRT_MODELS:

                print(f'Starting Analysis for task {task}, llm: {llm_model} and irt {irt_model_type}')
                expe_name = f"{llm_model}_hierar_irt_{irt_model_type}"

                item_response_analyzer = HierarchicalItemResponseModel(students=get_all_students(llm_model, task),
                                                                       irt_model_type=irt_model_type)
                estimator = item_response_analyzer.fit()
                all_stats[expe_name] = item_response_analyzer.compute_stats(estimator)

                item_response_analyzer.plot(estimator=estimator,
                                            exam_model=f'{task}:{llm_model.capitalize()}',
                                            save_path=f"{task_path}/12_{task}_fig_{expe_name}.png",
                                            font_size=12)

                item_response_analyzer.plot(estimator=estimator,
                                            exam_model=f'{task}:{llm_model.capitalize()}',
                                            save_path=f"{task_path}/14_{task}_fig_{expe_name}.png",
                                            font_size=14)

                item_response_analyzer.plot(estimator=estimator,
                                            exam_model=f'{task}:{llm_model.capitalize()}',
                                            save_path=f"{task_path}/16_{task}_fig_{expe_name}.png",
                                            font_size=16)

                item_response_analyzer.plot(estimator=estimator,
                                            exam_model=f'{task}:{llm_model.capitalize()}',
                                            save_path=f"{task_path}/18_{task}_fig_{expe_name}.png",
                                            font_size=18)

                item_response_analyzer.plot(estimator=estimator,
                                            exam_model=f'{task}:{llm_model.capitalize()}',
                                            save_path=f"{task_path}/20_{task}_fig_{expe_name}.png",
                                            font_size=20)

                item_response_analyzer.plot(estimator=estimator,
                                            exam_model=f'{task}:{llm_model.capitalize()}',
                                            save_path=f"{task_path}/22_{task}_fig_{expe_name}.png",
                                            font_size=22)

        with open(f"{task_path}/{task}_stats_hierar_irt.json", "w") as outfile:
            outfile.write(json.dumps(all_stats))

    for task in tqdm(TASKS):
        all_stats = {}
        task_path = f"{dirname(dirname(abspath(__file__)))}/Data/{task}/EvalResults/IRT"

        for llm_model in MODELS:

            for irt_model_type in [2, 3]:

                print(f'Starting Analysis for task {task}, llm: {llm_model} and irt {irt_model_type}')
                expe_name = f"{llm_model}_base_irt_{irt_model_type}"

                item_response_analyzer = ItemResponseModel(students=get_all_students(llm_model, task),
                                                           irt_model_type=irt_model_type)
                estimator = item_response_analyzer.fit()
                all_stats[expe_name] = item_response_analyzer.compute_stats(estimator)

                item_response_analyzer.plot(estimator=estimator,
                                            exam_model=f'{task}:{llm_model.capitalize()}',
                                            save_path=f"{task_path}/fig_{expe_name}.png")

        with open(f"{task_path}/stats_base_irt.json", "w") as outfile:
            outfile.write(json.dumps(all_stats))
