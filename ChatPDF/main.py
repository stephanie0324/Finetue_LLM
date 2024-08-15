# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader, PyPDFLoader
# from langchain.text_splitter import  RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain

import os


import uuid
import base64
from IPython import display
from unstructured.partition.auto import partition
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever


output_path = "./output/"

elements = partition("./data/2303.18223v13.pdf")
print("\n\n".join([str(el) for el in elements]))

# # Get elements
# raw_pdf_elements = partition_pdf(
#     filename="./data/2303.18223v13.pdf",
#     extract_images_in_pdf=True,
#     infer_table_structure=True,
#     chunking_strategy="by_title",
#     max_characters=4000,
#     new_after_n_chars=3800,
#     combine_text_under_n_chars=2000,
#     extract_image_block_output_dir=output_path,
# )
# # Get text summaries and table summaries
# text_elements = []
# table_elements = []

# text_summaries = []
# table_summaries = []

# summary_prompt = """
# Summarize the following {element_type}: 
# {element}
# """
# summary_chain = LLMChain(
#     llm=ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024), 
#     prompt=PromptTemplate.from_template(summary_prompt)
# )

# for e in raw_pdf_elements:
#     if 'CompositeElement' in repr(e):
#         text_elements.append(e.text)
#         summary = summary_chain.run({'element_type': 'text', 'element': e})
#         text_summaries.append(summary)

#     elif 'Table' in repr(e):
#         table_elements.append(e.text)
#         summary = summary_chain.run({'element_type': 'table', 'element': e})
#         table_summaries.append(summary)

# # Get image summaries
# image_elements = []
# image_summaries = []

# def encode_image(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode('utf-8')
    
# def summarize_image(encoded_image):
#     prompt = [
#         SystemMessage(content="You are a bot that is good at analyzing images."),
#         HumanMessage(content=[
#             {
#                 "type": "text", 
#                 "text": "Describe the contents of this image."
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{encoded_image}"
#                 },
#             },
#         ])
#     ]
#     response = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024).invoke(prompt)
#     return response.content
    
# for i in os.listdir(output_path):
#     if i.endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(output_path, i)
#         encoded_image = encode_image(image_path)
#         image_elements.append(encoded_image)
#         summary = summarize_image(encoded_image)
#         image_summaries.append(summary)

"""Another example"""
# # 讀取檔案
# ##TODO mutli pdf input
# file_path = "/home/B20711/private_projects/LLM_Projects/ChatPDF/data/2303.18223v13.pdf"
# loader = file_path.endswith(".pdf") and PyPDFLoader(file_path) or TextLoader(file_path)

# # 選擇 splitter 並將文字切分成多個 chunk 
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) 
# texts = loader.load_and_split(splitter)

# # 建立本地 db
# embeddings = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(texts, embeddings)

# # 對話 chain
# qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())
# chat_history = []
# while True:
#     query = input('\nQ: ') 
#     if not query:
#         break
#     result = qa({"question": query + ' (用繁體中文回答)', "chat_history": chat_history})
#     print('A:', result['answer'])
#     chat_history.append((query, result['answer']))