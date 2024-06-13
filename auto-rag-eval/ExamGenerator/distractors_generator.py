import argparse
import concurrent.futures
import json
import logging
import time
from datetime import datetime
from os.path import abspath, dirname
from typing import Dict, List

from ExamGenerator.utils import get_single_file_in_folder
from LLMServer.bedrock.claude_instant import ClaudeInstant
from LLMServer.bedrock.claude_v2 import ClaudeV2
from LLMServer.base_model import BaseLLM
from tqdm import tqdm

logger = logging.getLogger(__name__)
ROOTPATH = dirname(dirname(abspath(__file__)))


class LLMDistractorGenerator:

    def __init__(self,
                 llm_model: BaseLLM):

        self.llm_model = llm_model

    def make_distractor_prompts(self,
                                question: str,
                                answer: str) -> str:
        return (f"### Human: Here is  a technical question on AWS documentation: {question}."
                f"\nThe correct answer is {answer}.\nProvide 3 incorrect answers or distractors to "
                "this question.\n### Assistant:")

    def generate_distractors(self,
                             exam: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:

        generated_distractors = {}
        for k in tqdm(range(0, len(exam))):
            answer = self.llm_model.invoke(
                prompt=self.make_distractor_prompts(question=exam[k]['question'],
                                                    answer=exam[k]['correct_answer']),
                params={})
            generated_distractors[k] = {
                **exam[k],
                "raw_distractors": answer
            }
        return generated_distractors


class BatchDistractorGenerator:

    def __init__(self,
                 task_domain: str,
                 model_list: List[str],
                 batch_size: int):

        self.batch_size = batch_size
        self.model_list = model_list
        self.task_domain = task_domain

        self.model_map = {
            'ClaudeInstant': LLMDistractorGenerator(
                llm_model=ClaudeInstant()),
            'ClaudeV2': LLMDistractorGenerator(
                llm_model=ClaudeV2())
        }

    def batch_generate_distractors(self, exam_folder: str) -> None:

        with open(get_single_file_in_folder(exam_folder), "r") as f:
            data = json.load(f)

        logger.error((f"Processing a total of {len(data)} documentation pieces for {self.task_domain}"
                      f" using models {self.model_list}, with batch size of {self.batch_size} "
                      f"({1+len(data)//self.batch_size} batches)"))

        # Split the data into batches
        batches = [data[i:i + self.batch_size]
                   for i in range(0, len(data), self.batch_size)]

        start_time = datetime.fromtimestamp(
            time.time()).strftime('%Y%m%d%H')

        try:

            for batch_index, batch in enumerate(batches):
                logger.error(f"Running batch {batch_index}")

                with concurrent.futures.ProcessPoolExecutor() as executor:

                    futurs = {model: executor.submit(self.model_map[model].generate_distractors, batch)
                              for model in self.model_list}
                    updated_questions = {model: futur.result() for model, futur in futurs.items()}
                    # Write the dictionary to a JSON file
                    for model in updated_questions.keys():
                        filename = (f"{self.task_domain}_QCM_distractors_base_exam_{exam_folder.split('/')[-1]}"
                                    f"_to_{model}_{start_time}_batch{batch_index}.json")
                        with open(f"{ROOTPATH}/Data/{self.task_domain}/RawExamData/{filename}", "w") as write_file:
                            json.dump(updated_questions[model], write_file)

        except Exception as e:

            logger.error(f"Failure to generate disractors for batch {batch_index}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates Distractors from Exam Data")

    parser.add_argument(
        "--task-domain",
        help="Task Domain, among DevOps, StackExchange, MyOwnTask...",
    )

    parser.add_argument(
        "--exam-folder",
        help="Exam data to use to generate distractors, eg html_llamav2_2023091421...",
    )

    main_args, _ = parser.parse_known_args()

    raw_distractor_generator = BatchDistractorGenerator(batch_size=10,
                                                        task_domain=main_args.task_domain,
                                                        # model_list=['openllama', 'llamav2']
                                                        model_list=['openllama']
                                                        )

    # TODO: Modify prompt
    raw_distractor_generator.batch_generate_distractors(
        exam_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/ExamData/{main_args.exam_folder}")
