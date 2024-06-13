import json
import random
from os.path import abspath, dirname
from typing import Dict

ROOTPATH = dirname(dirname(abspath(__file__)))


class FakeExamGenerator:

    def __init__(self,
                 task_domain: str,
                 exam_folder: str):

        self.task_domain = task_domain
        self.exam_folder = exam_folder

    def generate_fake_distractor(self,
                                 exam_question: Dict[str, str]) -> Dict[str, str]:

        new_candidates = [") This is an absurd choice.",
                          ") This is an ridiculus choice.",
                          ") Picking this choice is a nonsense.",
                          exam_question['correct_answer'][1:]]

        # shuffle the candidates
        random.shuffle(new_candidates)

        # build new candidates list
        shuffled_candidates = [f"{chr(65 + i)}{x}"
                               for i, x in enumerate(new_candidates)]

        # find the new letter for the correct answer
        new_correct_answer = [f"{chr(65 + i)}{x}"
                              for i, x in enumerate(new_candidates) if x == exam_question['correct_answer'][1:]][0]

        return {**{k: v
                   for k, v in exam_question.items() if k not in ['choices', 'correct_answer']},
                'choices': shuffled_candidates,
                'correct_answer': new_correct_answer}

    def generate_fake_exam(self) -> None:

        with open(f"{ROOTPATH}/Data/{self.task_domain}/ExamData/{self.exam_folder}/exam.json", 'r') as f:
            self.exam_data = json.load(f)

        fake_exam = [self.generate_fake_distractor(exam_question=elem) for elem in self.exam_data]

        with open(f"{ROOTPATH}/Data/FakeExam/{self.task_domain}/fake_{self.exam_folder}.json", "w") as outfile:
            outfile.write(json.dumps(fake_exam))


if __name__ == '__main__':

    fake_exam = FakeExamGenerator(task_domain='Arxiv',
                                  exam_folder='llamav2_2023091905')

    fake_exam.generate_fake_exam()
