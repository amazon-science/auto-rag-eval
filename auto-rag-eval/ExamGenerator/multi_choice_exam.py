import argparse
import json
import logging
import os
import random
import re
from collections import Counter
from os.path import abspath, dirname
from typing import Dict, List

import numpy as np
from ExamGenerator.multi_choice_question import MultiChoiceQuestion
from ExamGenerator.utils import SimilarityChecker, get_n_sentences
from RetrievalSystems.bm25 import BM25ContextProvider
from RetrievalSystems.context_utils import ContextProvider
from RetrievalSystems.dpr_context_aggregator import DPRContextGenerator
from RetrievalSystems.embedding_retriever import EmbeddingContextProvider
from RetrievalSystems.siamese_retriever import SiameseContextProvider
from tqdm import tqdm

ROOTPATH = dirname(dirname(abspath(__file__)))
logger = logging.getLogger(__name__)


class MultiChoiceExam:

    def __init__(self,
                 task_domain: str,
                 model_name: str,
                 context_generator_dict: Dict[str, ContextProvider],
                 question_date: str):

        self.task_domain = task_domain
        self.model_name = model_name
        self.question_date = question_date
        self.context_generator_dict = context_generator_dict
        assert context_generator_dict is not None, "Provide Context Retriever Model"
        self.question_list: List[MultiChoiceQuestion] = []
        self.question_parsing_fail: int = 0
        self.choices_parsing_fail: int = 0
        self.correct_answer_parsing_fail: int = 0
        self.other_parsing_fail: int = 0
        self.failed_question_list: List[str] = []

    def load_from_list(self,
                       raw_exam_list: List[str]) -> None:

        for raw_question in raw_exam_list:

            mcq = MultiChoiceQuestion(documentation=raw_question['documentation']['text'],
                                      raw_answer=raw_question['answer'],
                                      model_name=self.model_name)

            mcq.extract_information()

            if mcq.valid_mcq() and self.task_based_constraints(mcq=mcq):
                mcq.add_retrieved_context(self.context_generator_dict)
                self.question_list.append(mcq)
            else:
                if mcq.question is None:
                    self.question_parsing_fail += 1

                if mcq.choices is None:
                    self.choices_parsing_fail += 1

                if mcq.correct_answer is None:
                    self.correct_answer_parsing_fail += 1

                if mcq.valid_mcq():
                    self.other_parsing_fail += 1

                self.failed_question_list.append(mcq.raw_answer)

    def load_all_model_question(self) -> bool:

        exam_directory = f"{ROOTPATH}/Data/{self.task_domain}/RawExamData/"
        self.n_question = 0

        logger.error(f"Starting to load all raw questions from {exam_directory}")

        raw_question_files = [os.path.join(exam_directory, f)
                              for f in os.listdir(exam_directory)
                              if (os.path.isfile(os.path.join(exam_directory, f))
                                  and f.startswith(f"{self.task_domain}_QCM_{self.model_name}_{self.question_date}"))]

        if len(raw_question_files) == 0:

            return False

        for file in tqdm(raw_question_files):

            with open(file, "r") as f:
                raw_exam_list = list(json.load(f).values())
                self.load_from_list(raw_exam_list=raw_exam_list)
                self.n_question += len(raw_exam_list)

        return True

    def task_based_constraints(self,
                               mcq: MultiChoiceQuestion) -> bool:

        def refers_to_document(question: str) -> bool:
            # Patterns prioritizing specificity
            document_patterns = [
                # term immediately followed by title in quotes
                r'\b(documentation|paper|article|research|study)\b\s*\"[^\"]+\"',
                # citation-like sentence followed by title
                r'\b(discussed in|addressed in|described in|of the)\b\s*\"[^\"]+\"',
                # fallback to original terms
                r'\b(documentation|paper|article|research|study)\b',
            ]

            # Check if any of the patterns match
            for pattern in document_patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return False
            return True

        if self.task_domain in ['Arxiv', 'StackExchange']:

            return refers_to_document(mcq.question)

        else:

            return True

    def compute_exam_analytics(self,
                               save_failed_question: bool,
                               display_n_samples: int = 1) -> None:

        if self.n_question == 0:
            raise ValueError("Empty exam, please check model name, date and path to ensure the exam is loaded properly.")

        if save_failed_question:

            with open((f"{ROOTPATH}/ExamGenerator/DebugingData/failed_questions_"
                       f"{self.task_domain}_{self.model_name}_{self.question_date}.json"), "w") as outfile:
                outfile.write(json.dumps(self.failed_question_list))

        def convert_perc(x):
            return 100 * x / len(self.question_list)

        logger.error(
            f"\n###################### {self.model_name} ######################\n")
        logger.error(f"ExamID: {self.task_domain}:{self.model_name}:{self.question_date}")
        logger.error(("\n### Parsing Analysis:\n\n"
                      f"Total of {len(self.question_list)}/{self.n_question} questions processed"
                      f" ({100*len(self.question_list)/self.n_question:.02f}%)"))
        logger.error((f"Statistics over {len(self.failed_question_list)} failed parsing:\n"
                      f"Question Parsing Error: {100*self.question_parsing_fail/len(self.failed_question_list):.02f}%\n"
                      f"Choices Parsing Error: {100*self.choices_parsing_fail/len(self.failed_question_list):.02f}%\n"
                      f"Correct Answer Parsing Error: {100*self.correct_answer_parsing_fail/len(self.failed_question_list):.02f}%\n"
                      f"Other Parsing Error or Constraints: {100*self.other_parsing_fail/len(self.failed_question_list):.02f}%\n"))

        if len(self.question_list) == 0:
            raise ValueError(
                "None of the questions are properly parsed. Please check the parsing logic, using for instance ExamAnalysis/failed_question_analysis.ipynb")

        # Positional bias has been removed so
        # ---
        answer_analysis = Counter([question.correct_answer[0]
                                   for question in self.question_list])
        # logger.error(
        #    (f"Position Bias: {answer_analysis} (Baseline Acc: {100*max(answer_analysis.values())/len(self.question_list):.02f}%)"))

        logger.error(("### Accuracy Analysis:\n\n"
                      f"Best Fixed Answer Baseline: {convert_perc(max(answer_analysis.values())):.02f}%\n"
                      f"Longest Answer Baseline: {convert_perc(sum([mcq.correct_candidate_is_longest() for mcq in self.question_list])):.02f}%\n"))

        logger.error("### Sample questions:\n\n{}".format('\n'.join([f"Question {k+1}: {mcq.question}"
                                                                     for k, mcq in enumerate(self.question_list[:10])])))

        question_keyword = ['Which', 'What', 'How', 'When', 'Why', 'Where']
        question_counter = [(f"{k}{(7-len(k))*' '} -- "
                             f"{convert_perc(sum([k.lower() in mcq.question.lower() for mcq in self.question_list])):.02f}%")
                            for k in question_keyword]
        other_key = sum([not (any([k.lower() in mcq.question.lower() for k in question_keyword]))
                         for mcq in self.question_list])
        question_counter.append(f"Other   -- {convert_perc(other_key):.02f}%")

        logger.error("\n### Question Analysis\n")
        logger.error("Question type:\n{}".format('\n'.join(question_counter)))
        first_word_analysis = sum([mcq.question.split(' ')[0].lower() in [e.lower() for e in question_keyword]
                                   for mcq in self.question_list])
        logger.error(f"\nQuestion starts with {question_keyword}: {convert_perc(first_word_analysis):.02f}%")
        logger.error(("Avg. question char. length: "
                      f"{np.mean([len(mcq.question) for mcq in self.question_list]):.02f}"
                      f" (std: {np.std([len(mcq.question) for mcq in self.question_list]):.02f})"))
        logger.error((f"Avg. number of sentence in question: "
                      f"{np.mean([get_n_sentences(mcq.question) for mcq in self.question_list]):.02f}"))
        logger.error(("Avg. answers char. length: "
                      f"{np.mean([len(''.join(mcq.choices))/4 for mcq in self.question_list]):.02f}"
                      f" (std: {np.std([len(''.join(mcq.choices))/4 for mcq in self.question_list]):.02f})"))
        logger.error(("Avg. correct answer char. length: "
                      f"{np.mean([len(mcq.correct_answer) for mcq in self.question_list]):.02f}"
                      f" (std: {np.std([len(mcq.correct_answer) for mcq in self.question_list]):.02f})"))
        logger.error(("Avg. documentation char. length: "
                      f"{np.mean([len(mcq.documentation) for mcq in self.question_list]):.02f}"
                      f" (std: {np.std([len(mcq.question) for mcq in self.question_list]):.02f})"))
        logger.error(("Avg. number of sentence in documentation: "
                      f"{np.mean([get_n_sentences(mcq.documentation) for mcq in self.question_list]):.02f}\n"))

        for elem in random.sample(self.question_list, display_n_samples):

            similarity_checker = SimilarityChecker()

            elem.display(similarity_checker)

    def save_exam_dataset(self) -> None:

        if self.n_question == 0:
            raise ValueError("Empty exam, please check model name, date and path to ensure the exam is loaded properly.")

        docs_exam = [{'question': mcq.question,
                      'documentation': mcq.documentation,
                      'choices': mcq.choices,
                      'correct_answer': mcq.correct_answer,
                      'retrieved_context': mcq.retrieved_context,
                      } for mcq in self.question_list]

        dir_path = f"{ROOTPATH}/Data/{self.task_domain}/ExamData/{self.model_name}_{self.question_date}"
        os.makedirs(dir_path,
                    exist_ok=True)

        with open(f"{dir_path}/exam.json", "w") as outfile:
            outfile.write(json.dumps(docs_exam))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Creates Exam from Raw Exam Data")

    parser.add_argument(
        "--task-domain",
        help="Task Domain, among DevOps, StackExchange...",
    )
    parser.add_argument(
        "--question-date",
        help="Date associated with the raw exam (eg 2023091223), can be seen in RawExamData",
    )
    parser.add_argument('--save-exam',
                        action='store_true',
                        help='If provided, the exam is saved. Otherwise, we just compute analytics')

    main_args, _ = parser.parse_known_args()

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

    for model_name in ['llamav2', 'openllama', 'claudev2', 'claude_instant']:

        MultiChoiceExamLLM = MultiChoiceExam(task_domain=main_args.task_domain,
                                             model_name=model_name,
                                             question_date=main_args.question_date,
                                             context_generator_dict=context_generator_dict)

        llm_exam_exists = MultiChoiceExamLLM.load_all_model_question()

        if llm_exam_exists:

            MultiChoiceExamLLM.compute_exam_analytics(save_failed_question=True)

            if main_args.save_exam:
                MultiChoiceExamLLM.save_exam_dataset()
