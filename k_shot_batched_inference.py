import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--start", type=int, help="start index")
parser.add_argument("--end", type=int, help="end index")
parser.add_argument("--device", type=str, help="device_id (cuda:0)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inferencing")
parser.add_argument("--k_times", type=int, default=8, help="Number of times to ask each question")
parser.add_argument("--gsm_test_file_path", type=str, default="gsm_eval_set.csv", help="Path to the GSM test file")

args = parser.parse_args()

start = args.start
end = args.end
device = args.device
batch_size = args.batch_size
k_times = args.k_times
model_name = args.model
gsm_test_file_path = args.gsm_test_file_path

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = "right"
model.to(device)

# Load the file
df = pd.read_csv(gsm_test_file_path)
questions = df['question'].to_list()
questions = questions[start:end]

def batch_inference(model, tokenizer, batch_questions, device, k):
    outputs = []
    for _ in range(k):
        test_prompts = [f"[INST] {q} [/INST] " for q in batch_questions]
        model_inputs = tokenizer(test_prompts, return_tensors='pt', padding=True).to(device)
        greedy_outputs = model.generate(**model_inputs, max_new_tokens=1500, do_sample=True, temperature=0.3)
        outputs.append([tokenizer.decode(greedy_output, skip_special_tokens=True) for greedy_output in greedy_outputs])
    return outputs

# List to hold all the result dataframes
result_dataframes = []

# Process questions in batches
for i in range(0, len(questions), batch_size):
    batch_questions = questions[i:i+batch_size]
    print("*" * 50)
    print(f"Processing questions {i} to {i + len(batch_questions) - 1}")
    batch_outputs = batch_inference(model, tokenizer, batch_questions, device, k_times)

    # Transpose the batch outputs to get a list of answers for each question
    batch_outputs_transposed = list(map(list, zip(*batch_outputs)))

    # Create a dataframe for the current batch and add it to the list
    batch_df = pd.DataFrame({'question': batch_questions})
    for j in range(k_times):
        batch_df[f'answer{j+1}'] = [answers[j] for answers in batch_outputs_transposed]
    result_dataframes.append(batch_df)

# Concatenate all the result dataframes
result_df = pd.concat(result_dataframes, ignore_index=True)

# Save results to CSV
result_df.to_csv(f'GSM_Evaluation_{start}_{end}.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script execution time: {elapsed_time} seconds")