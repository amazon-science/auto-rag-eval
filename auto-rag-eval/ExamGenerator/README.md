# Exam Generation

This folder contains several function and notebook utilies for the automated generation of the exam.

## Generation

* `question_generator.py`: Python class to generate the raw exam, given a knowledge corpus.
* `multi_choice_question.py`: Python classes `MultiChoiceQuestionParser` and `MultiChoiceQuestion` to convert raw question into filtered questions.
* `multi_choice_exam.py`: Python class `MultiChoiceExam` to be invoked to generate the processed exam from the raw exam data.

## Distractors

* `distractors_generator.py` can be used to generate new distractors for your exam questions, following a two-step approach.

## Utilities

* `fake_exam_generator.py`: Code to create a fake exam from a given exam to check the validity of the evaluation.
* `utils.py`: Among others, class `SimilarityChecker` to evaluate the similarity between answers, using embeddings and n-gram based methods.
* In case you already generated an exam and want to add new retriever to evaluate, one can use the code from `extend_ir_existing_exam.py`
