# Information Retirval
## What is IR?
The process of information retrieval involves **indexing documents**, **receiving user queries**, **searching for relevant content in the indexed data**, **ranking the results**, and presenting the most pertinent information to the user.
From IR to RAG &rarr; [more information](https://aman.ai/primers/ai/RAG/)
<style>
    .center-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
</style>

<img src="https://miro.medium.com/v2/resize:fit:1358/1*VRI5JHC9bhpDcfbWJYV9AQ.png" alt="Description of the image" class="center-image" width = 400>


####　Closed-Form Generative QA System (封閉型生成式QA系統):

* Focuses on **using a single knowledge source**, often a large language model (LLM).
* Google's research on a closed-form generative QA system involves a Transformer model and a differentiable search index (DSI), which outperformed mainstream IR models.
* **Challenges** include ***addressing model hallucination***, ***verifying generated answers***, and ***expanding the model's knowledge base***.

#### Open-Form Generative QA System (開放型生成式QA系統):

* Involves integrating language models with external tools, such as **retrieval engines**, to enhance information retrieval and question-answering.
* Examples include Google's REALM and Meta AI's RAG system, which **use retrieval mechanisms to supplement generative models**.
* DeepMind's RETRO combines learning and retrieval modules with a large corpus of over a trillion tokens, improving knowledge completeness and retrieval accuracy.

## Implementation
