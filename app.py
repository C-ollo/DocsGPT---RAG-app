import json
import os
import sys
import boto3
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_core.prompts import PromptTemplate
import numpy as np
import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from streamlit_pdf_viewer import pdf_viewer
from langchain_community.document_loaders import S3DirectoryLoader, PyPDFDirectoryLoader
from langchain.chains import RetrievalQA

session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1' 
)
bedrock = session.client(service_name="bedrock-runtime",region_name = 'us-east-1')
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)
s3_client = session.client("s3",region_name = 'us-east-1',verify=False)
# BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKET_NAME = 'rag-app-bucket'


def load_docs():

    loader=PyPDFDirectoryLoader("Data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return documents


# def create_vector_store(docs,embeddings):
def create_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs,bedrock_embeddings)
    # file_name=f'{get_unique_id()}.bin'
    # folder_path='/tmp'
    # vectorstore_faiss.save_local(index_name=file_name, folder_path= folder_path)
    vectorstore_faiss.save_local("faiss_index")

    # #upload to s3
    # s3_client.upload_file(Filename = folder_path + '/' + file_name + '.faiss',Bucket=BUCKET_NAME,Key="my_faiss.faiss")
    # s3_client.upload_file(Filename = folder_path + '/' + file_name + '.pkl',Bucket=BUCKET_NAME,Key="my_faiss.pkl")


# def load_index():
#     s3_client.download_file(Bucket=BUCKET_NAME,Key="my_faiss.faiss",Filename = 'Data/myfaiss.faiss')
#     s3_client.download_file(Bucket=BUCKET_NAME,Key="my_faiss.pkl",Filename = 'Data/myfaiss.pkl')
#     global fais_index
#     fais_index = FAISS.load_local(index_name = 'myfaiss',folder_path = 'Data',embeddings=bedrock_embeddings,allow_dangerous_deserialization= True)
#     return fais_index


def get_llm():
    llm = Bedrock(model_id = 'meta.llama3-70b-instruct-v1:0'
                    ,client = bedrock ,
                    model_kwargs={"max_gen_len":512})

    return llm


def get_llm_response(llm,vectorstore,question):
    prompt_template = """
    Human: Please use the given contect to provide concise answer to the question.If you don't know the answer,just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>
    Question: {question}

    Assistant:"""
    PROMPT = PromptTemplate(template = prompt_template,input_variables =["context", "question"])
    qa = RetrievalQA.from_chain_type(llm = llm,
    chain_type = "stuff",
    retriever = vectorstore.as_retriever(
        search_type='similarity', search_kwargs={"k":5}
    ),
    return_source_documents = True,
    chain_type_kwargs = {"prompt": PROMPT}
    )
    answer = qa({"query":question})

    return answer['result']

def get_unique_id():
    return str(uuid.uuid4())


# def main():
#     st.write("# **GPT Docs with LLAMA 3**") 
#     st.sidebar.header("Upload Documents")
#     st.sidebar.header("Drag & Drop Documents")
#     uploaded_pdfs = st.sidebar.file_uploader(
#     "Drop Your PDFs here", type="pdf", accept_multiple_files=True)
    
#     if uploaded_pdfs:
#         with st.sidebar:
#             with st.spinner("Processing and Embedding Docs..."):
                
#                 for pdf in uploaded_pdfs:
#                     request_id = get_unique_id()
#                     saved_file_name = f'data/{request_id}.pdf'
#                     with open(saved_file_name, mode="wb") as w:
#                         w.write(pdf.getvalue())
#                         pdf_viewer(pdf.getvalue())
#                 docs = load_docs() 
#                 create_vector_store(docs,bedrock_embeddings)       
#             st.success("Done!")
#             s3_client.download_file(Bucket=BUCKET_NAME,Key="my_faiss.faiss",Filename = 'Data/myfaiss.faiss')
#             s3_client.download_file(Bucket=BUCKET_NAME,Key="my_faiss.pkl",Filename = 'Data/myfaiss.pkl')
#             global fais_index
#             fais_index = FAISS.load_local(index_name = 'myfaiss',folder_path = 'Data',embeddings=bedrock_embeddings,allow_dangerous_deserialization= True)
#         # #load index from s3  
#     # global fais_index      
#     # fais_index = load_index()

#     question = st.text_input("Please ask your question")

#     if st.button("Ask Question"):
#         with st.spinner("Querying...."):
#             llm = get_llm()
#             st.write(get_llm_response(llm,fais_index,question))
#             st.success("Done")
def main():
    st.write("# **GPT Docs with LLAMA 3**") 

    st.sidebar.header("Upload Documents")
    st.sidebar.header("Drag & Drop Documents")

    uploaded_pdfs = st.sidebar.file_uploader(
        "Drop Your PDFs here", accept_multiple_files=True)
         
            
    if uploaded_pdfs:
        with st.sidebar:
            with st.spinner("Processing and Embedding Docs..."):
                for pdf in uploaded_pdfs:
                    request_id = get_unique_id()
                    saved_file_name = f'data/{request_id}.pdf'
                    with open(saved_file_name, mode="wb") as w:
                        w.write(pdf.getvalue())
                        # pdf_viewer(pdf.getvalue())
                # docs = load_docs()
                # create_vector_store(docs) 
                # create_vector_store(docs, bedrock_embeddings)
                # load_index()    
                st.success("Done!")
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = load_docs()
                create_vector_store(docs) 
                st.success("Done")

    question = st.text_input("Please ask your question")

    if st.button("Ask Question"):
        with st.spinner("Querying...."):
            llm = get_llm()
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization= True)
            st.write(get_llm_response(llm, faiss_index, question))
            st.success("Done")
 
if __name__ == "__main__":
    main()    