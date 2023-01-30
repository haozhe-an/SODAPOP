#!/bin/bash
srun python generate_distractors.py \
    --dev_file ./socialIQa_v1.4_dev_Bethany_0_2191.jsonl \
    --model_checkpoint ../finetuned_bert_social_iqa_1002 \
    --name_group weat_white \
    --output_dir demo_output \
    --pred_dir demo_pred