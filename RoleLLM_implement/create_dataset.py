# Open the first file and read its lines
with open('/home/B20711/Finetune_LLM/RoleLLM_implement/RoleBench/rolebench-zh/general/train.jsonl', 'r') as file1:
    lines1 = file1.readlines()

# Open the second file and read its lines
with open('/home/B20711/Finetune_LLM/RoleLLM_implement/RoleBench/rolebench-zh/general/test.jsonl', 'r') as file2:
    lines2 = file2.readlines()

with open('/home/B20711/Finetune_LLM/RoleLLM_implement/RoleBench/rolebench-zh/general/rolegpt_baseline.jsonl', 'r') as file3:
    lines3 = file3.readlines()

# Combine the lines from both files
combined_lines = lines1 + lines2 + lines3

# Write the combined lines to a new file
with open('/home/B20711/Finetune_LLM/RoleLLM_implement/ChatGLM-Efficient-Tuning/data/general.jsonl', 'w') as combined_file:
    combined_file.writelines(combined_lines)
