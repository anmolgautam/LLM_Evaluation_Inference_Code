# Large Language Model Inference

This repository contains code for performing inference using a large language model (LLM) on a set of questions. The script takes a CSV file with questions, runs them through the model, and generates answers for each question multiple times.

## Features

- Batch processing of questions for efficient inference
- Multiple generations per question to explore different possible answers
- Easy to use command-line interface for setting up the inference parameters

## Requirements

- Python 3.x
- pandas
- transformers
- torch

## Setup

Install the required Python packages using:

```bash
pip install pandas transformers torch
```

## Usage

Run the script from the command line, providing the necessary arguments:

```bash
python inference_script.py --model <model_name> --start <start_index> --end <end_index> --device <device_id> --batch_size <batch_size> --k_times <k_times> --gsm_test_file_path <file_path>
```

### Arguments

- `--model`: Name or path of the pre-trained model to use for inference.
- `--start`: Start index of the questions to process from the CSV file.
- `--end`: End index of the questions to process from the CSV file.
- `--device`: Device to run the inference on (e.g., 'cuda:0' for GPU).
- `--batch_size`: Number of questions to process in each batch (default: 8).
- `--k_times`: Number of times to generate an answer for each question (default: 8).
- `--gsm_test_file_path`: Path to the CSV file containing the questions (default: 'gsm_eval_set.csv').

## Output

The script will generate a CSV file named `GSM_Evaluation_<start>_<end>.csv` containing the original questions and their respective generated answers.

## Example

```bash
python inference_script.py --model gpt2 --start 0 --end 100 --device cuda:0 --batch_size 10 --k_times 5 --gsm_test_file_path questions.csv
```

This command will process questions from index 0 to 100 using the `gpt2` model, with a batch size of 10, generating 5 answers per question, and running on the GPU (device 'cuda:0').

## Performance

The script logs the execution time at the end of the process.

---

Please ensure that you have the necessary computational resources to run the inference, as large language models can be resource-intensive.
