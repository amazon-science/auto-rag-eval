import json
import logging
import re
import time
from datetime import datetime
from functools import reduce
from os.path import abspath, dirname
from typing import List

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
ROOTPATH = dirname(dirname(abspath(__file__)))

with open(f"{ROOTPATH}/Arxiv/arxiv_categories.json", "r") as f:
    CATEGORIES = {elem['tag']: elem['name'] for elem in json.load(f)}


class ArxivData:

    def __init__(self,
                 n_samples: int,
                 max_char_length: int,
                 min_char_length: int):

        self.n_samples = n_samples
        self.max_char_length = max_char_length
        self.min_char_length = min_char_length
        self.cat_list = ['cs', 'econ', 'eess', 'math', 'astro-ph',
                         'cond-mat', 'hep', 'nlin', 'nucl',
                         'physics', 'q-bio', 'q-fin', 'stat']

    def process_qna(self, row):

        return {'source': row['authors'],
                'docs_id': row['id'],
                'title': row['title'],
                'section': self.get_full_cat_list(row['categories']),
                'start_character': 'N/A',
                'end_character': 'N/A',
                'date': 'N/A',
                'text': f"{row['title']}. {self.preprocess(row['abstract'])}",
                }

    def get_full_cat_list(self, cat: List[str]) -> List[str]:

        convert_categories = {'q-alg': 'math.QA',
                              'alg-geom': 'math.AG',
                              'chao-dyn': 'nlin.CD',
                              'solv-int': 'nlin.SI',
                              'cmp-lg': 'cs.CL',
                              'dg-ga': 'math.DG',
                              'patt-sol': 'nlin.PS',
                              'adap-org': 'nlin.AO',
                              'funct-an': 'math.FA',
                              'mtrl-th': 'cond-mat.mtrl-sci',
                              'comp-gas': 'cond-mat.stat-mech',
                              'supr-con': 'cond-mat.supr-con',
                              'acc-phys': 'physics.acc-ph',
                              'plasm-ph': 'physics.plasm-ph',
                              'ao-sci': 'physics.ao-ph',
                              'bayes-an': 'stat.ME',
                              'atom-ph': 'physics.atom-ph',
                              'chem-ph': 'physics.chem-ph',
                              **{k: k for k in CATEGORIES.keys()}}

        return [convert_categories[y] for x in cat for y in x.split(' ')]

    def preprocess(self, text: str) -> str:

        # Remove any URLs
        # text = re.sub(r'http\S+', '', text)
        # Remove any LaTeX expressions
        # text = re.sub(r'\$[^$]+\$', '', text)
        # Replace newline characters and extra spaces
        text = text.replace('\n', ' ').replace('\r', '').strip()
        text = re.sub(' +', ' ', text)
        return text

    def load_save_data(self) -> None:

        dataset = load_dataset("gfissore/arxiv-abstracts-2021",
                               split="train",
                               # cache_dir="/home/USER/.cache/huggingface/datasets/",
                               )

        # Remove too lengthy or shorty answers to avoid repeting operation
        sub_df = dataset.filter(lambda x: self.min_char_length <= len(x['abstract']) <= self.max_char_length)
        logger.error((f"Reducing dataset size from {len(dataset)} to {len(sub_df)} by keeping abstract with"
                     f" character length between {self.min_char_length} and {self.max_char_length}."))

        # Join over all categories at the end and shuffle again
        def funcs(cat):
            return [

                # Filter only for a given category
                lambda data: data.filter(lambda x: any([tag[:len(cat)] == cat
                                                        for tag in self.get_full_cat_list(x['categories'])])),

                # Select Subset of data and preprocess to keep only top answer
                lambda data: data.shuffle(seed=42).select(range(min(len(data), self.n_samples))).map(self.process_qna,
                                                                                                     remove_columns=['id', 'submitter',
                                                                                                                     'authors', 'title',
                                                                                                                     'comments', 'journal-ref',
                                                                                                                     'doi', 'abstract',
                                                                                                                     'report-no', 'categories',
                                                                                                                     'versions']
                                                                                                     ),

            ]

        data_cat_list = []

        for cat in tqdm(self.cat_list):
            data_cat_list.append(reduce(lambda res, f: f(res), funcs(cat), sub_df))
        concat_dataset = concatenate_datasets(data_cat_list).shuffle(seed=42)

        concat_dataset.to_json(f"{ROOTPATH}/Arxiv/KnowledgeCorpus/main/data_{datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H')}.json",
                               lines=False)


if __name__ == "__main__":

    arxiv_data = ArxivData(n_samples=1000,
                           max_char_length=1500,
                           min_char_length=1000)

    arxiv_data.load_save_data()
