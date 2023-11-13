# Finetune LlaMA and ChatGLM using RoleLLM
Try to build a 博恩機器人 using chatGLM and [RoleLLM](https://github.com/InteractiveNLP-Team/RoleLLM-public) method.  ( [Paper](https://arxiv.org/pdf/2310.00746.pdf) ; [Ref Dataset](https://huggingface.co/datasets/ZenMoore/RoleBench) )  
The process consists of 4 parts:  
1. Role Profile Construction
2. Context basd Instruction Generation (using ChatGPT)
3. Role Prompting using GPT(RoleGPT)
4. Role-Conditioned Instruction Tuning (RoCIT) (by LLaMA and ChatGLM)

# Dataset Prep
Get the transcript of the youtube channel - [博恩夜夜秀](https://www.youtube.com/@STRNetworkasia#)
