#!/bin/bash
#Task supported: StackExchange, DevOps and rag variants StackExchangeRag, DevOpsRag...

cd lm-evaluation-harness
current_date=$(date +"%y%m%d%H")
task_domain=$1
echo "Evaluating ${task_domain} task"

model_path="add/you/model/path/here"
echo "Evaluating Llamav2 - 13B - ICL@0"
accelerate launch main.py \
    --model hf \
    --model_args "pretrained=${model_path},load_in_8bit=True" \
    --tasks "${task_domain}Exam" \
    --device cuda \
    --output_path "results/${task_domain}Exam/llamav2/13b/results_${current_date}_icl0.json"
echo "Evaluating Llamav2 - 13B - ICL@1"
accelerate launch main.py \
    --model hf \
    --model_args "pretrained=${model_path},load_in_8bit=True" \
    --tasks "${task_domain}Exam" \
    --device cuda \
    --num_fewshot 1 \
    --output_path "results/${task_domain}Exam/llamav2/13b/results_${current_date}_icl1.json"
echo "Evaluating Llamav2 - 13B - ICL@2"
accelerate launch main.py \
    --model hf \
    --model_args "pretrained=${model_path},load_in_8bit=True" \
    --tasks "${task_domain}Exam" \
    --device cuda \
    --num_fewshot 2 \
    --output_path "results/${task_domain}Exam/llamav2/13b/results_${current_date}_icl2.json"

# Note the difference in arguments when using 70B models: python3 + parallelize=True vs accelerate launch
model_path="add/you/model/path/here"
echo "Evaluating Llamav2 - 70B - ICL@0"
python3 main.py \
    --model hf \
    --model_args "pretrained=${model_path},parallelize=True" \
    --tasks "${task_domain}Exam" \
    --device cuda \
    --output_path "results/${task_domain}Exam/llamav2/70b/results_${current_date}_icl0.json"
echo "Evaluating Llamav2 - 70B - ICL@1"
python3 main.py \
    --model hf \
    --model_args "pretrained=${model_path},parallelize=True" \
    --tasks "${task_domain}Exam" \
    --device cuda \
    --num_fewshot 1 \
    --output_path "results/${task_domain}Exam/llamav2/70b/results_${current_date}_icl1.json"
echo "Evaluating Llamav2 - 70B - ICL@2"
python3 main.py \
    --model hf \
    --model_args "pretrained=${model_path},parallelize=True" \
    --tasks "${task_domain}Exam" \
    --device cuda \
    --num_fewshot 2 \
    --output_path "results/${task_domain}Exam/llamav2/70b/results_${current_date}_icl2.json"