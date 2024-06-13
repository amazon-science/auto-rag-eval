import json
from os.path import abspath, dirname

from ExamAnalysis.item_response_models import ExamSetting
from ExamAnalysis.iterative_item_response_models import \
    IterativeHierarchicalItemResponseModel
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
         ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/llamav2/13b/full_sample_{task}Exam_rag_multi_qa_{model}_results_*_new_ir_icl{i}.jsonl',
                     llm='llamav2:13B',
                     retrieval='rag_multi_qa',
                     icl=i,
                     name=f'Rag MultiQA@{i} [13B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/llamav2/13b/full_sample_{task}Exam_rag_dpr_bm25_multi_qa_{model}_results_*_new_ir_icl{i}.jsonl',
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
        ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/llamav2/70b/full_sample_{task}Exam_rag_multi_qa_{model}_results_*_new_ir_icl{i}.jsonl',
                     llm='llamav2:70B',
                     retrieval='rag_multi_qa',
                     icl=i,
                     name=f'Rag MultiQA@{i} [70B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/llamav2/70b/full_sample_{task}Exam_rag_dpr_bm25_multi_qa_{model}_results_*_new_ir_icl{i}.jsonl',
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
        ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/mistral/7b/full_sample_{task}Exam_rag_multi_qa_{model}_results_*_new_ir_icl{i}.jsonl',
                     llm='mistral:7b',
                     retrieval='rag_multi_qa',
                     icl=i,
                     name=f'Rag MultiQA@{i} [7B]'),
         ExamSetting(path_pattern=f'{root_path}/{task}NewRagExam/mistral/7b/full_sample_{task}Exam_rag_dpr_bm25_multi_qa_{model}_results_*_new_ir_icl{i}.jsonl',
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

    LLM_MODELS = ["llamav2"]
    TASKS = ['DevOps', 'StackExchange', 'Arxiv', 'SecFilings']
    IRT_TYPE = [3]
    N_STEPS = 4
    DROP_RATIO = 0.1

    for task in tqdm(TASKS):
        all_stats = {}
        task_path = f"{dirname(dirname(abspath(__file__)))}/Data/{task}/EvalResults/IterativeIRT"

        for llm_model in LLM_MODELS:

            for irt_model_type in IRT_TYPE:

                print(f'Starting Analysis for task {task}, llm: {llm_model} and irt {irt_model_type}')
                expe_name = f"{llm_model}_recursive_irt_{irt_model_type}"

                iterative_item_response_analyzer = IterativeHierarchicalItemResponseModel(students=get_all_students(llm_model, task),
                                                                                          irt_model_type=irt_model_type)
                estimator_dict = iterative_item_response_analyzer.fit(n_steps = N_STEPS,
                                                                      drop_ratio = DROP_RATIO)
                all_stats[expe_name] = {step_k: iterative_item_response_analyzer.compute_stats(estimator_dict[step_k]) 
                                        for step_k in estimator_dict.keys()}

                iterative_item_response_analyzer.plot_iterative_informativeness(
                    estimator_dict=estimator_dict,
                    exam_model=f'{task}:{llm_model.capitalize()}',
                    save_path=f"{task_path}/18_{task}_fig_{expe_name}_step{N_STEPS}.png")

        with open(f"{task_path}/recursive_irt_step{N_STEPS}.json", "w") as outfile:
            outfile.write(json.dumps(all_stats))