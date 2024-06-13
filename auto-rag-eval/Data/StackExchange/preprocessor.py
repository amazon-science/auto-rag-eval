import re
import time
from datetime import datetime
from functools import reduce
from os.path import abspath, dirname
from typing import Dict, List

from datasets import concatenate_datasets, load_dataset
from markdownify import markdownify as md
from tqdm import tqdm

ROOTPATH = dirname(dirname(abspath(__file__)))


# Please use first the HuggingFace script at https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences to get the data

class StackExchangeData:

    def __init__(self,
                 n_samples: int,
                 max_char_length: int):

        self.n_samples = n_samples
        self.max_char_length = max_char_length
        self.cat_list = ['Stackoverflow', 'math', 'superuser',
                         'serverfault', 'askubuntu', 'electronics',
                         'physics', 'unix', 'tex', 'english',
                         'meta', 'apple', 'ell', 'gaming',
                         'stats', 'softwareengineering',
                         'mathoverflow', 'gis', 'diy', 'magento']

    def get_best_answer(self, answers: List[Dict[str, str]]):
        """return the best answer, that is, the one with the highest score"""
        best_index = 0
        best_score = answers[0]["pm_score"]
        for i in range(1, len(answers)):
            if answers[i]["pm_score"] > best_score :
                best_score = answers[i]["pm_score"]
                best_index = i
        return answers[best_index]["text"]

    def lang_callback(self, el):
        lang = el['class'][0] if el.has_attr('class') else None
        return lang.split("-")[-1] if lang else None

    def html2md(self, text: str) -> str:
        text = md(text, code_language_callback=self.lang_callback)
        text = re.sub(r"\n\s*\n", "\n\n", text).strip()
        return text.encode('utf-8', 'replace').decode()

    def process_qna(self, row):

        return {'source': row['metadata'],
                'docs_id': row['qid'],
                'title': 'N/A',
                'section': 'N/A',
                'start_character': 'N/A',
                'end_character': 'N/A',
                'text': self.html2md(f"### User: {row['question']} -\n\n### Top Answer: {self.get_best_answer(row['answers'])}"),
                }

    def get_topic(self, source_list: List[str]) -> str:

        filtered_list = list(set([elem.replace('https://', '').split('/')[0].split('.')[0] for elem in source_list]))

        return filtered_list[0]

    def load_save_dataset(self) -> None:

        # Heavy dataset, ~22GB, use cache location if not enough memory
        dataset = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                               split="train",
                               # cache_dir="/home/USERID/.cache/huggingface/datasets/",
                               )

        # funcs = [
        #     # Select Subset of data and preprocess to keep only top answer
        #     lambda data: data.shuffle(seed=42).select(range(self.n_samples)).map(self.process_qna,
        #                                                                          remove_columns=['qid', 'metadata', 'answers', 'question']
        #                                                                          ),
        #     # Remove too lengthy answers
        #     lambda data: data.filter(lambda x: len(x['text']) <= self.max_char_length)
        # ]

        # filtered_dataset = reduce(lambda res, f: f(res), funcs, dataset)
        # filtered_dataset.to_json(f"{ROOTPATH}/Data/StackExchange/KnowledgeCorpus/main/data_{datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H')}.json",
        #                          lines=False)

        # Join over all categories at the end and shuffle again
        def funcs(cat):
            return [

                # Filter only for a given category
                lambda data: data.filter(lambda x: cat == self.get_topic(x['metadata'])),

                # Select Subset of data and preprocess to keep only top answer
                lambda data: data.shuffle(seed=42).select(range(min(len(data), self.n_samples))).map(self.process_qna,
                                                                                                     remove_columns=['qid',
                                                                                                                     'metadata',
                                                                                                                     'answers',
                                                                                                                     'question']
                                                                                                     ),

            ]

        data_cat_list = []

        for cat in tqdm(self.cat_list):
            data_cat_list.append(reduce(lambda res, f: f(res), funcs(cat), dataset))
        concat_dataset = concatenate_datasets(data_cat_list).shuffle(seed=42)

        concat_dataset.to_json(f"{ROOTPATH}/Data/StackExchange/KnowledgeCorpus/main/data_{datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H')}.json",
                               lines=False)


if __name__ == "__main__":

    stack_exchange_data = StackExchangeData(
        n_samples=400,
        max_char_length=1500)

    stack_exchange_data.load_save_dataset()
