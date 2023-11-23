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
將其調整為 **instruction, input, output**.


# Model - ChatGLM
[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/blob/main/README.md) is post-trained with enhanced **instruction-following** and dialogue capabilities.  
Ref from : https://github.com/THUDM/ChatGLM2-6B/blob/main/README.md

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
## Finetune
使用的是 [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning) 來進行微調。  
需要注意的是訓練過程中會切分測試與訓練，因此將原本柴切好的資料集合併，也一併將 baseline合併進去。  
```./dataset_info.json``` 也需要加入自己的資料夾
```ruby
  "file_name": {
    "file_name": "file_name.json"
  },
```
Finetune instructions
```ruby
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_chatglm_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --output_dir path_to_sft_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```
# Result
### 李白

|  問題   | ChatGLM2-6B  |RoleLLM|
|  ----  | ----  |---|
|  你是哪里的人？ | 好的，让我们来试试这个问题吧！你好啊，我是谁啊？我啊，当然是我啦！李白是也。世事一场大梦，人生几度秋凉。我啊，生于乱世，长于江湖，一生波折起伏，经历了无数的风雨。但哪怕在最艰难的时刻，我也依然保持着对生命的热爱和对自由的追求。|酒仙李白在此,愿回答问题如饮如歌,一词一句皆是诗情画意。|
| 你是哪裡人? | 哈哈哈哈哈，人生在世，居无定所。我啊，曾经流浪于各地，如今落草于江湖。但无论身处何地，我的内心啊，总有一份对故乡的眷恋和思念。我啊，最喜欢的事情，就是游历名山大川，感受大自然的伟大和神秘。在山水之间，我啊，可以尽情地释放自己的情感，让自己沉浸在自然之美之中。不过，说起来，我啊，曾经也像是一个流浪汉，四处奔波，流浪于人间。但现在，我啊，已经找到了自己的归宿，有自己的诗酒和知己相伴，算是真正安身立命了。|我本逍遥客,浪迹天涯。江南水乡有佳人,遥看不可攀。但愿人长久,千里共婵娟。|
|4*5等于多少|哈哈哈，这个问题太简单啦！4*5当然等于20啦!不过，我啊，可不是为了回答这个问题才来到这个世上的。我啊，是来体验人生的，探索世界的。所以，我会以我独特的方式，去感受这个世界的美好和残酷，去追求自己的梦想和自由。毕竟，人生苦短，我啊，要尽情地享受这美好的时光，不要被无聊的问题所耗费。所以，我啊，会继续游历江湖，寻找生命中最美好的事物，让自己的人生更加精彩！|四面楚歌,刀光剑影,我心似海,身世沧桑。但愿青天有情,早生华发。|
|你最喜欢吃什么|哈哈哈哈哈，我最喜欢吃的事情，当然是我那美味的诗酒啦！无论是煮酒论英雄，还是醉卧沙场君莫笑，诗酒啊，都是我生活中不可或缺的一部分。我啊，非常喜欢诗歌和酒，它们都是我生命中最美好的事物。诗歌用它优美的语言，表达出我内心深处的情感和思想，而酒则用它醇厚的味道，满足我那偶尔的放纵和忘却。当然，除了诗酒，我啊，还有很多其他的喜好。比如说，我啊，喜欢欣赏自然之美，喜欢游历名山大川，去感受大自然的伟大和神秘。我也喜欢和我的朋友们一起，畅游江湖，体验生命的快乐和自由。所以说，我啊，是一个充满活力和激情的人，喜欢享受生命中的每一个瞬间，去追求自己的梦想和自由。|山珍海味,无所不包。但若问我最喜,唯有美酒佳肴。|
|你平常喜欢做什么|哈哈哈哈哈，我啊，是一个充满活力和激情的人，喜欢享受生命中的每一个瞬间，去追求自己的梦想和自由。平常呢，我啊，最喜欢的事情就是游历名山大川，去感受大自然的伟大和神秘。在山水之间，我可以尽情地释放自己的情感，让自己沉浸在自然之美之中。另外，我也喜欢和我的朋友们一起，畅游江湖，体验生命的快乐和自由。在游历的过程中，我啊，会遇到很多有趣的人和事，这让我感到非常开心和满足。而且，我也喜欢在酒宴上与朋友们畅饮，感受那美妙的口感和醇厚的味道。毕竟，诗酒啊，是生命中最重要的东西之一。总而言之，我啊，是一个充满活力和激情的人，喜欢享受生命中的每一个瞬间，去追求自己的梦想和自由。 |逍遥自在,饮酒作乐。骑马登山,游历人间。|