import logging
from os.path import abspath, dirname
from typing import Dict, List

from tqdm import tqdm
from LLMServer.base_model import BaseLLM

logger = logging.getLogger(__name__)
ROOTPATH = dirname(dirname(abspath(__file__)))


class LLMExamGenerator:

    def __init__(self,
                 step_size: int,
                 task_domain: str,
                 llm_model: BaseLLM):

        # Step size is to mitigate when one model inference is faster than another
        # eg openllama:13b = 3* llamav2:70B
        self.step_size = step_size
        self.task_domain = task_domain
        self.llm_model = llm_model

    def make_question_prompt(self, documentation: str) -> str:
        # Adding the syntax constraint was done in V2 and appears to impact the formatting of the question.
        return (f"### Human: Here is some documentation from {self.task_domain}: {documentation}.\n"
                "From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
                " 1 correct answer and explanations. Syntax should be Question: {question}\nA){candidate A}\nB){candidate B}\n"
                "C){candidate C}\nD){candidate D} Correct Answer: {correct answer}\n### Assistant:")

    # def make_question_prompt_icl(self, example, documentation: str) -> str:
    #     # icl = (f"### Human: Here is some documentation from {self.task_domain}: {example.documentation}.\n"
    #     #        f"From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
    #     #        " 1 correct answer and explanations.\n### Assistant:"
    #     #        "Question: {}\nCandidates: {}\n".format(example.question, '\n'.join(example.choices))
    #     #        f"Correct Answer: {example.correct_answer}\n")
    #     prompt = (f"### Human: Here is some documentation from {self.task_domain}: {documentation}.\n"
    #               f"From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
    #               " 1 correct answer and explanations.\n### Assistant:")
    #     return f"{icl}\n{prompt}"

    def generate_exam(self, data: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:

        generated_questions = {}
        for k in tqdm(range(0, len(data), self.step_size)):
            answer = self.llm_model.invoke(
                prompt=self.make_question_prompt(data[k]['text']),
                params={})
            generated_questions[k] = {
                "documentation": data[k],
                "answer": answer
            }
        return generated_questions


class ClaudeExamGenerator(LLMExamGenerator):

    def __init__(self,
                 step_size: int,
                 task_domain: str,
                 llm_model: BaseLLM):

        super().__init__(step_size=step_size,
                         task_domain=task_domain,
                         llm_model=llm_model)

    def make_question_prompt(self, documentation: str) -> str:
        return (f"\n\nHuman: Here is some documentation from {self.task_domain}: {documentation}.\n"
                "From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
                " 1 correct answer and explanations. Syntax should be Question: {question}\nA){candidate A}\nB){candidate B}\n"
                "C){candidate C}\nD){candidate D} Correct Answer: {correct answer}\n\nAssistant:")

    def generate_exam(self, data: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:

        generated_questions = {}
        for k in tqdm(range(0, len(data), self.step_size)):
            answer = self.llm_model.invoke(
                prompt=self.make_question_prompt(data[k]['text']),
                params={})
            generated_questions[k] = {
                "documentation": data[k],
                "answer": answer
            }
        return generated_questions
