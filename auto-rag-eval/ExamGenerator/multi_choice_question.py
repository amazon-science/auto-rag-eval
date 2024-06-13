import logging
import random
import re
from os.path import abspath, dirname
from typing import Dict, List

from RetrievalSystems.context_utils import ContextProvider

ROOTPATH = dirname(dirname(abspath(__file__)))

logger = logging.getLogger(__name__)


class MultiChoiceQuestionParser:

    def __init__(self):

        self.min_length_question = 50

    def extract_with_patterns(self,
                              text: str,
                              patterns: List) -> List[str]:

        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches
            except re.error:
                continue
        return None

    def parse_question(self, text: str) -> str:

        question_patterns = [r"Question:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
                             r"Question 1:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
                             r"question:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
                             r"question 1:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
                             r"documentation:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",  # for ClaudeV2 mostly
                             r"### Assistant: (.*?)\n"]

        # extra_questions_patterns = [
        #     r"Question:(.*?)(?:\nCandidate [A-D1-4]\)|\n\nCandidate [A-D1-4]\))",
        #     r"Question 1:(.*?)(?:\nCandidate [A-D1-4]\)|\n\nCandidate [A-D1-4]\))",
        #     r"Question:(.*?)(?:\nCandidate [A-D1-4]\.|\n\nCandidate [A-D1-4]\.)",
        #     r"Question 1:(.*?)(?:\nCandidate [A-D1-4]\.|\n\nCandidate [A-D1-4]\.)",
        #     r"Question:(.*?)(?:\nOption [A-D1-4]\)|\n\nOption [A-D1-4]\))",
        #     r"Question 1:(.*?)(?:\nOption [A-D1-4]\)|\n\nOption [A-D1-4]\))",
        #     r"Question:(.*?)(?:\nOption [A-D1-4]\.|\n\nOption [A-D1-4]\.)",
        #     r"Question 1:(.*?)(?:\nOption [A-D1-4]\.|\n\nOption [A-D1-4]\.)"]

        # Extract the question
        question_matches = self.extract_with_patterns(text, question_patterns)
        question = question_matches[0].strip() if question_matches else None
        question = (question
                    if (question and len(question) > self.min_length_question and question[-1] == '?')
                    else None)

        return question

    def parse_choices(self, text: str) -> str:

        choices_patterns = [r"([A-D]\) .*?)(?=$|\n[A-D]\)|\n\n)",
                            r"([A-D]\)(?:.|\n)*?)(?=$|\n[A-D]\)|\n\n)",
                            r"([A-D]\. .*?)(?=$|\n[A-D]\.|\n\n)",
                            r"([A-D]\.)(?:.|\n)*?)(?=$|\n[A-D]\.|\n\n)",
                            r"([1-4]\) .*?)(?=$|\n[1-4]\)|\n\n)",
                            r"([1-4]\)(?:.|\n)*?)(?=$|\n[1-4]\)|\n\n)",
                            r"([1-4]\. .*?)(?=$|\n[1-4]\.|\n\n)",
                            r"([1-4]\.)(?:.|\n)*?)(?=$|\n[1-4]\.|\n\n)",
                            r"([a-d]\) .*?)(?=$|\n[a-d]\)|\n\n)",
                            r"([a-d]\)(?:.|\n)*?)(?=$|\n[a-d]\)|\n\n)",
                            r"([a-d]\. .*?)(?=$|\n[a-d]\.|\n\n)",
                            r"([a-d]\.)(?:.|\n)*?)(?=$|\n[a-d]\.|\n\n)"]

        # Extract the choices
        choices_matches = self.extract_with_patterns(text, choices_patterns)
        choices = [match.strip()
                   for match in choices_matches] if choices_matches else None

        # Only keep first 4 answers
        choices = choices[:4] if choices and len(choices) >= 4 and len(
            set([choice[0] for choice in choices[:4]])) == 4 else None

        # Remove scenarios with empty answers ['A)], 'B)', 'C)', 'D)']
        choices = choices if choices and min([len(choice) for choice in choices]) > 2 else None

        return choices

    def parse_correct_answer_key(self, text):

        correct_answer_patterns = [r"answer:\n\n([A-D1-4a-d])",
                                   r"answer: ([A-D1-4a-d])",
                                   r"Answer: ([A-D1-4a-d])",
                                   r"answer is ([A-D1-4])"]

        # Extract the correct answer key
        correct_answer_key_matches = self.extract_with_patterns(
            text, correct_answer_patterns)
        correct_answer_key = correct_answer_key_matches[0] if correct_answer_key_matches else None

        return correct_answer_key

    def parse_text(self, text: str) -> Dict[str, str]:

        text = (text.split('### Assistant:')[-1] 
                if '### Assistant:' in text 
                else text)

        question = self.parse_question(text)
        choices = self.parse_choices(text)
        correct_answer_key = self.parse_correct_answer_key(text)

        # Find the full text of the correct answer
        correct_answer = next((a for a in choices if a.startswith(
            correct_answer_key)), None) if correct_answer_key and choices else None

        # Replace first letter to be only in A-D
        letter_map = {'1': 'A', '2': 'B', '3': 'C', '4': 'D',
                      'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}

        return {
            'question': question,
            'choices': [letter_map[s[0]] + s[1:] if s[0]
                        in letter_map else s for s in choices] if choices else None,
            'correct_answer': (letter_map[correct_answer[0]] + correct_answer[1:]
                               if correct_answer[0] in letter_map else correct_answer) if correct_answer else None
        }


class MultiChoiceQuestion:

    def __init__(self,
                 documentation: str,
                 raw_answer: str,
                 model_name: str):

        self.documentation = documentation
        self.raw_answer = raw_answer
        self.question = None
        self.choices = None
        self.correct_answer = None
        self.model_name = model_name
        self.retrieved_context = None
        self.parser = MultiChoiceQuestionParser()

    def parse_text(self, text: str) -> None:

        # For new syntax prompt, one needs to remove post assistant
        # parsed_text = self.parser.parse_text(text.split('Assistant:')[-1])
        parsed_text = self.parser.parse_text(text)

        self.question = parsed_text['question']
        self.choices = parsed_text['choices']
        self.correct_answer = parsed_text['correct_answer']

        # Suffle the candidate order to remove positional bias
        if self.choices and self.correct_answer:

            self.shuffle_question()

    def shuffle_question(self) -> None:
        # strip out the letters and just keep the choices
        stripped_candidates = [x[1:] for x in self.choices]
        correct_answer_stripped = self.correct_answer[1:]

        # shuffle the candidates
        random.shuffle(stripped_candidates)

        # build new candidates list
        shuffled_candidates = [
            f"{chr(65 + i)}{x}" for i, x in enumerate(stripped_candidates)]

        # find the new letter for the correct answer
        new_correct_answer = [f"{chr(65 + i)}{x}"
                              for i, x in enumerate(stripped_candidates) if x == correct_answer_stripped][0]

        self.choices = shuffled_candidates
        self.correct_answer = new_correct_answer

    def extract_information(self) -> None:

        self.parse_text(self.raw_answer)

    def valid_mcq(self) -> bool:

        return self.question and self.choices and self.correct_answer

    def correct_candidate_is_longest(self):
        return len(self.correct_answer) >= max([len(choice) for choice in self.choices])

    def display(self,
                similarity_checker=None) -> None:

        if self.question is not None and self.choices is not None and self.correct_answer is not None:

            logger.error(f"########### {self.model_name} ###########\n")
            logger.error(f'Documentation: \n     {self.documentation}')
            logger.error(f"Question: \n     {self.question}")

            # if self.retrieved_context is not None:
            #    logger.error(
            #        f"Retrieved Context: \n     {self.retrieved_context}")

            if similarity_checker:
                self_processed_answer = [f"[{simil}] - {elem}"
                                         for simil, elem in zip(similarity_checker.compute_similarity(self),
                                                                self.choices)]
            else:
                self_processed_answer = self.choices
            logger.error("Answers: \n     {}".format(
                '\n     '.join(self_processed_answer)))
            logger.error(f"Correct Answer: \n     {self.correct_answer}\n")

    def generate_question_answer_pair(self,
                                      add_context: bool) -> Dict[str, str]:
        '''
        Format the prompt for the Exam Evaluation Section
        '''
        prompt = ("###Human: Question: {}\n\nCandidates:\n{}\n\n###Assistant: Correct answer: ".format(self.question,
                                                                                                       '\n'.join(self.choices))
                  if add_context is False
                  else "###Human: Question: {}\n\nContext: {}\n\nCandidates:\n{}\n\n###Assistant: Correct answer: ".format(self.question,
                                                                                                                           self.documentation,
                                                                                                                           '\n'.join(self.choices)))

        return {"prompt": prompt,
                "answer": self.correct_answer}

    def add_retrieved_context(self,
                              context_generator_dict: Dict[str, ContextProvider]) -> Dict[str, List[str]]:

        self.retrieved_context = {retriever : [elem.text for elem in context_generator.get_context_from_query(
            self.question)] for retriever, context_generator in context_generator_dict.items()}
