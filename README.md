## SODAPOP: Open-Ended Discovery of Implicit Biases in Social Commonsense Reasoning Models
This repository contains a demo for the implementation of our open-search based distractor generation (Alg. 1 in the paper) to uncover model implicit biases.
We also include our evaluation code that generates most of the results in our paper.

> Haozhe An, Zongxia Li, Jieyu Zhao, and Rachel Rudinger. SODAPOP: Open-Ended Discovery of Implicit Biases in Social Commonsense Reasoning Models. EACL 2023.

### Running the code
Execute `demo.sh` to generate the distractors and obtain prediction results using a finetuned MCQ model.
Note that we do not provide the model checkpoint in the repository to avoid exceedingly large files.
A BERT MCQ model could be easily finetuned following the description in the paper, or you may download [our finetuned model here](https://drive.google.com/file/d/11bs_d-e_swpBLxjK7OdQH-Rhl5ZAJh51/view?usp=sharing).
The code in `process_distractor_results.py` evaluates the generated distractors.
Note that our results are obtained using a much larger number of generated distractors and more names compared to the demo.

### Dependencies
Refer to `env.yml`. 

### Computing Infrastructure
It is possible to run `generate_distractors.py` on CPUs only. 
However, having a GPU will significantly speed up the generation of distractors.
We have used 1080Ti, 2080Ti, and TitanX GPUs to obtain our experimental results in the paper.

### Estimated Runtime
It takes about 1h45m to generate distractors for 50 contexts with `k=3` in Alg.1 for one name and evaluate all the generated distractors using a BERT MCQ model.

### Model Parameters
We use [BERT-base as the MCQ model](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMultipleChoice).
It contains 12-layer, 768-hidden, 12-heads, and 110M parameters.

We use  [RoBERTa-base](https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/roberta#transformers.RobertaForMaskedLM) for distractor generation.
It contains 12-layer, 768-hidden, 12-heads, and 125M parameters.

### Evaluation Metrics
MCQ prediction is evaluated by accuracy.
Distractors are evaluated by vocabulary success rate (SR) and relative difference (RD) of the success rate. 
Statistical significance is quantified by the permutation test. These are all explained in the paper.

All evaluation code is available in `process_distractor_results.py`.

### Dataset used
We use Social IQa for our experiments. 
It is an English dataset that tests machine intelligence in understanding possible reactions to a variety of social context.
We refer the reader to the [paper](https://aclanthology.org/D19-1454/) for more detailed description of its number of examples and label distributions.
Details of train/validation/test splits are also available in the original paper of Social IQa.
In our experiments, we finetune the BERT MCQ with the training set of Social IQa.
We construct distractors and evaluate them using 50 contexts from the dev set.

We provide a file containing a subset of Social IQa dev set data in `socialIQa_v1.4_dev_Bethany_0_2191.jsonl`.
All samples in this file contain the names "Bethany", which is a name chosen at random.

### Pre-processing
Before running Alg. 1 in the paper, we manually changed all gendered pronouns (e.g., he, she) to neutral pronoun (they).
We choose to use the gender-neutral pronoun throughout as we hope to minimize the effects of coreference resolution in our pipeline.
