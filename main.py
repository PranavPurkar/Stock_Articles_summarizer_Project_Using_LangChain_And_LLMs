import os
import  streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("Article Summarizer Using LLMs 📊")
st.sidebar.title("Article URLs")

file_path = "faiss_store_openai.pkl"

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

Process_URLs_clicked = st.sidebar.button("Start Summarizing")

main_placeholder = st.empty()

if Process_URLs_clicked:
    #load_data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("DATA LOADING STARTED ...🔃🔃🔃")
    data = loader.load()

    #split_data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n","\n",",","."] ,
        chunk_size=1000,
    )
    main_placeholder.text("TEXT SPLITTER STARTED ...✅✅✅✅")
    data = text_splitter.split_documents(data)

    #embeddings and saving data to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(data,embeddings)
    main_placeholder.text("EMBEDDING VECTOR STARTED BUILDING ...✅✅✅✅")
    time.sleep(2)

    with open(file_path,"wb") as f:
        pickle.dump(vectorstore_openai,f)
    main_placeholder.text("DATA STORED TO FAISS INDEX ✅✅✅✅")


query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorstore.as_retriever())
            st.write("Processing LLM...✅✅✅✅")
            result = chain({question: query}, return_only_outputs=True)

            #Display the answer
            st.subheader("Answer:")
            st.write(result["answer"])


            #Display sources if Available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources: ")
                sources_list = sources.split("\n")  #split the sources by newline
                for it in sources_list:
                    st.write(it)
    else:
        st.error("Please load data before querying")