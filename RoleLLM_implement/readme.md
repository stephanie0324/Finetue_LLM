# Finetune LlaMA and ChatGLM using RoleBench with LoRA
Try to build a 博恩機器人 using [RoleLLM](https://github.com/InteractiveNLP-Team/RoleLLM-public) method.  ( [Paper](https://arxiv.org/pdf/2310.00746.pdf) ; [Ref Dataset](https://huggingface.co/datasets/ZenMoore/RoleBench) )  
### RoCIT / Context Instruct
The process consists of 4 parts:  
```
1. Role Profile Construction
2. Context basd Instruction Generation  
   * Using ChatGPT to generate high quality QA pairs from profiles to extract role-specific knowledge 
3. Role Prompting using GPT(RoleGPT)  

----------------------------- ( 1~3 : Create RoleBench Dataset ) ----------------

4. Role-Conditioned Instruction Tuning (RoCIT)
    * Fine-tune ChatGLM on RoleBench -> RoleGLM
```
# Dataset
這裡一共有兩種做法，作者提供的就是現成的資料集，而另一種是我們自己蒐集的博恩講稿，目前還在調整中。
* 博恩資料集 : Get the transcript of the youtube channel - [博恩夜夜秀](https://www.youtube.com/@STRNetworkasia#) (目前還在整理資料集) 
* RoleBench provided by authors
Dataset Link : [RoleBench](https://huggingface.co/datasets/ZenMoore/RoleBench/tree/main/rolebench-zh/general)  

## RoleBench 介紹與準備
RoleBench dataset中分別有 general / specific :
* general : 一般問答
* specific : 角色故事背景相關問題  

原先Rolebench dataset中資料格式如下，分別為 **role**, **question**, **generated**：
```
{
    "role": "孙悟空",
    "question": 编辑以下句子中的语法错误：他们去得机场太晚，结果错过了飞机。",
    "generated": "["哈哈，俺老孙看来这句话应该这么说：“他们去机场去得太晚，结果错过了飞机。”", "咳咳，现代人，你这句子应该这样说：他们去机场去得太晚，结果错过了飞机。", "哎呀，这句话说得不对劲，应该是\"他们去机场去得太晚，结果错过了飞机\"。", "哦，那句子应该是“他们去机场去得太晚，结果错过了飞机。”", "哈哈，这还用得着我美猴王指点么？改过来就是：他们去机场去得太晚，结果错过了飞机。"]"
}
```
將instruct與input合併 做preprocessing
```
{
    "prompt": "role": "孙悟空"\n input: 编辑以下句子中的语法错误：他们去得机场太晚，结果错过了飞机。", 
    "target": "["哈哈，俺老孙看来这句话应该这么说：“他们去机场去得太晚，结果错过了飞机。”", "咳咳，现代人，你这句子应该这样说：他们去机场去得太晚，结果错过了飞机。", "哎呀，这句话说得不对劲，应该是\"他们去机场去得太晚，结果错过了飞机\"。", "哦，那句子应该是“他们去机场去得太晚，结果错过了飞机。”", "哈哈，这还用得着我美猴王指点么？改过来就是：他们去机场去得太晚，结果错过了飞机。"]`"
}
```


# Model - ChatGLM
[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/blob/main/README.md) is post-trained with enhanced **instruction-following** and dialogue capabilities.  
Ref from : https://github.com/THUDM/ChatGLM2-6B/blob/main/README.md
## Pre-requisites
* Python : python3.8 up
```
git clone https://github.com/THUDM/ChatGLM2-6B
cd ChatGLM2-6B

pip install -r requirements.txt

# 為了降低模型存取大小
pip install accelerate
pip install bitsandbytes
```
## Model use
```ruby
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM2-6B,很高兴见到你,欢迎问我任何问题。
>>> response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
>>> print(response)
晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:

1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。
2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。
3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。
4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。
5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。
6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。

如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。
```

# Result
