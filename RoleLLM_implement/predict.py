
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('/home/B20711/Finetune_LLM/RoleLLM_implement/instruction.txt','r')as f:
    instructions = f.read()
# print(instructions)


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().to(device)
model = model.eval()
response, history = model.chat(tokenizer,instructions+'你是谁？', history=[])
print('Response: ',response,'\n','-'*30)
response, history = model.chat(tokenizer,instructions+'你住哪里?', history=history)
print('Response: ',response,'\n','-'*30)
response, history = model.chat(tokenizer,instructions+'4*5等于多少', history=history)
print('Response: ',response,'\n','-'*30)
response, history = model.chat(tokenizer,instructions+'你最喜欢吃什么', history=history)
print('Response: ',response,'\n','-'*30)
response, history = model.chat(tokenizer,instructions+'你平常喜欢做什么', history=history)
print('Response: ',response,'\n','-'*30)

print()
print('='*50)
print(f'After Training ............')
print('='*50)
print()

finetune_model_path = '/home/B20711/Finetune_LLM/RoleLLM_implement/output'
finetune_model = PeftModel.from_pretrained(model, finetune_model_path).half().to(device)
finetune_model = finetune_model.eval()
response, history = finetune_model.chat(tokenizer,instructions+'你是谁？', history=[])
print('Response: ',response,'\n','-'*30)
response, history = finetune_model.chat(tokenizer,instructions+'你住哪里?', history=history)
print('Response: ',response,'\n','-'*30)
response, history = finetune_model.chat(tokenizer,instructions+'4*5等于多少', history=history)
print('Response: ',response,'\n','-'*30)
response, history = finetune_model.chat(tokenizer,instructions+'你最喜欢吃什么', history=history)
print('Response: ',response,'\n','-'*30)
response, history = finetune_model.chat(tokenizer,instructions+'你平常喜欢做什么', history=history)
print('Response: ',response,'\n','-'*30)

del model
del finetune_model
torch.cuda.empty_cache()



