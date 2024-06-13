# Exam Evaluation

We leverage [lm-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) package to evaluate the (LLM&Retrieval) system on the generated exam.
To do, follow the next steps (lm-harness usage might have been updated since):

### Create a benchmark

Create a benchmark folder for for your task, here `DevOpsExam`, see `ExamEvaluator/DevOpsExam` for the template.
It contains a code file preprocess_exam,py for prompt templates and more importantly, a set of tasks to evaluate models on:

* `DevOpsExam` contains the tasks associated to ClosedBook (not retrieval) and OpenBook (Oracle Retrieval).
* `DevOpsRagExam` contains the tasks associated to Retrieval variants (DPR/Embeddings/BM25...).

The script`task_evaluation.sh` provided illustrates the evalation of `Llamav2:Chat:13B` and `Llamav2:Chat:70B` on the task, using In-Context-Learning (ICL) with respectively 0, 1 and 2 samples.
