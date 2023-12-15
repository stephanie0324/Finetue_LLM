# Retrieval-Augmented Generation (RAG)

## What is RAG ?
The core idea behind RAG is to utilize a retriever to select relevant passages or documents from a large corpus of text and then use a language model to generate responses based on the retrieved information. This approach aims to overcome limitations in solely using end-to-end generative models by leveraging the benefits of retrieval.
<p style="text-align: center">
    <img src="https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/jumpstart/jumpstart-fm-rag.jpg"alt="image" width="500" height="auto">
    <figcaption align='center' >Flow of RAG</figcaption>
</p>

* **Retriever**: This component is responsible for selecting relevant passages or documents from a large corpus of text. It helps identify contextually relevant information for a given query.
* **Generator**: This component is a large language model, often based on transformer architectures like BERT or T5. It generates responses based on the information retrieved by the retriever, aiming to produce coherent and contextually appropriate answers.

## Objective
* We try to build a generative QA model using data from 全國工商行政服務網常見FAQ​. 
* Users are allowed to ask questions in free form and we match the most similar QA pair and return the answer.

## Requirements
```ruby
python==3.10 up
openai==1.2.0
langchain==0.0.332
```

## Result
