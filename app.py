import streamlit as st 
from pandasai.llm.openai import OpenAI
import os
import pandas as pd
from pandasai import PandasAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import tempfile



openai_api_key ="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"


def chat_with_csv(df,prompt):
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = PandasAI(llm)
    result = pandas_ai.run(df, prompt=prompt)
    print(result)
    return result

st.set_page_config(layout='wide')

st.title("ChatCSV powered by LLM")


    
    
file = st.file_uploader("Upload your CSV file", type=['csv','pdf'])


if file is not None:
    file_extension = file.name.split('.')[-1]
    
    if file_extension == 'csv':

        col1, col2 = st.columns([1,1])
        with col1:
           st.info("CSV Uploaded Successfully")
           data = pd.read_csv(file)
           st.dataframe(data, use_container_width=True)
        with col2:
            st.info("Chat Below")
             
            input_text = st.text_area("Enter your query")

            if input_text is not None:
                if st.button("Chat with CSV"):
                    st.info("Your Query: "+input_text)
                    result = chat_with_csv(data, input_text)
                    st.success(result)
        
        
    elif file_extension == 'pdf':
        st.success("Uploaded file is a PDF.")
        if file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
               
            # load document
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            # split the documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            # select which embeddings we want to use
            embeddings = OpenAIEmbeddings(openai_api_key="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")
            # create the vectorestore to use as the index
            db = Chroma.from_documents(texts, embeddings)
            # expose this index in a retriever interface
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
            # create a chain to answer questions
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
            
            col1, col2 = st.columns([1,1])
            with col1:
                st.info("Chat Below")
                 
                input_text = st.text_area("Enter your query")

                if input_text is not None:
                    if st.button("Chat with PDF"):
                        st.info("Your Query: "+input_text)
                        result = result = qa({"query": input_text})
                        st.success(result['result'])
            
            
    else:
        st.warning("Uploaded file has an unsupported extension.")

        
