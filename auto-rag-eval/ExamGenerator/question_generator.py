import argparse
import concurrent.futures
import json
import logging
import random
import time
from datetime import datetime
from os.path import abspath, dirname
from typing import List

from ExamGenerator.utils import get_single_file_in_folder
from LLMServer.bedrock.claude_instant import ClaudeInstant
from LLMServer.bedrock.claude_v2 import ClaudeV2
from LLMServer.llm_exam_generator import ClaudeExamGenerator, LLMExamGenerator

logger = logging.getLogger(__name__)
ROOTPATH = dirname(dirname(abspath(__file__)))


class BatchExamGenerator:

    def __init__(self,
                 task_domain: str,
                 model_list: List[str],
                 batch_size: int):

        self.batch_size = batch_size
        self.model_list = model_list
        self.task_domain = task_domain

        self.model_map = {
                          'claudev2': ClaudeExamGenerator(step_size=1,
                                                          task_domain=self.task_domain,
                                                          llm_model=ClaudeV2()),
                          'claude_instant': ClaudeExamGenerator(step_size=1,
                                                                task_domain=self.task_domain,
                                                                llm_model=ClaudeInstant())
                          }
        assert not (any([model not in self.model_map.keys() for model in self.model_list]))

    def batch_generate_exam(self, data_folder: str) -> None:

        with open(get_single_file_in_folder(data_folder), "r") as f:
            data = json.load(f)

        # Suffle the data to prevent overfocusing on a topic
        # ---
        random.seed(10)
        random.shuffle(data)

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

                logger.error(f"Running batch {batch_index}.")
                if len(self.model_list) > 1:
                    # Multiprocessing not compatible with Bedrock Usage
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        futurs = {model: executor.submit(self.model_map[model].generate_exam, batch)
                                  for model in self.model_list}
                        generated_questions = {model: futur.result() for model, futur in futurs.items()}
                else:
                    generated_questions = {model: self.model_map[model].generate_exam(batch)
                                           for model in self.model_list}

                # Write the dictionary to a JSON file
                for model in generated_questions.keys():
                    filename = f"{self.task_domain}_QCM_{model}_{start_time}_batch{batch_index}.json"
                    with open(f"{ROOTPATH}/Data/{self.task_domain}/RawExamData/{filename}", "w") as write_file:
                        json.dump(generated_questions[model], write_file)

        except Exception as e:

            logger.error(f"Failure to collect questions for batch {batch_index}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates Raw Exam from Documentation Corpus")

    parser.add_argument(
        "--task-domain",
        help="Task Domain, among DevOps, StackExchange, MyOwnTask...",
    )

    main_args, _ = parser.parse_known_args()

    raw_exam_generator = BatchExamGenerator(batch_size=60,
                                            task_domain=main_args.task_domain,
                                            # model_list=['openllama', 'llamav2']
                                            model_list=['claudev2']
                                            )

    raw_exam_generator.batch_generate_exam(
        data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main")
