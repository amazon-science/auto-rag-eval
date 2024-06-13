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


# SEC TEMPLATE 

# [KEEP] 0.Business: Overview of the company's main operations, including its products or services.
# [KEEP] 1.Risk Factors: Discussion of risks and challenges the company faces.
# [REMOVE] 2.Unresolved Staff Comments: Comments by SEC staff on the company's previous filings that haven't been resolved.
# [REMOVE] 3.Properties: Information about the company's physical properties (like real estate).
# [REMOVE] 4.Legal Proceedings: Information on any significant legal actions involving the company.
# [REMOVE] 5.Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities: Details about the company’s stock, including dividends, the number of shareholders, and any buyback programs.
# [REMOVE] 6.Selected Financial Data: Summary of specific financial data for a five-year period.
    
# [KEEP] 8.Management’s Discussion and Analysis of Financial Condition and Results of Operations (MD&A): A detailed analysis from management’s perspective on the company’s financials and operations.
# [REMOVE] 9.Quantitative and Qualitative Disclosures About Market Risk: Information on market risk, such as foreign exchange risk, interest rate risk, etc.
# [REMOVE] 1.Financial Statements and Supplementary Data: Complete financial statements including balance sheets, income statements, and cash flow statements.
# [REMOVE] 11.Changes in and Disagreements with Accountants on Accounting and Financial Disclosure: If there have been changes or disagreements with accountants, this section provides details.
# [REMOVE] 12.Directors, Executive Officers and Corporate Governance: Information about the company’s directors and high-level executives.
# [REMOVE] 13.Executive Compensation: Detailed information about the compensation of top executives.
# [REMOVE] 14.Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters: Details about the shares held by major shareholders and company executives.
# [REMOVE] 15.Certain Relationships and Related Transactions, and Director Independence: Information about any transactions between the company and its directors or executives.
# [REMOVE] 16.Principal Accountant Fees and Services: Fees and services provided by the company's accountants.
# [REMOVE] 17.Exhibits, Financial Statement Schedules: Lists all the exhibits and financial statements schedules.
# [REMOVE] 18.Form 10-K Summary: Summary of the key information from the 10-K (optional).
# [REMOVE] 19. [OPTIONAl] CEO and CFO Certifications: As required by the Sarbanes-Oxley Act, certifications by the CEO and CFO regarding the accuracy of the financial statements.


class ExchangeData:

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

    stack_exchange_data = ExchangeData(n_samples=400,
                                       max_char_length=1500)

    stack_exchange_data.load_save_dataset()
