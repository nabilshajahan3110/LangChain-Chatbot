import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile
import os

st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("📄 Chat with your PDF (Local RAG Chatbot)")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    # Load PDF content
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embedding and FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # Load local model
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # RAG chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User query
    query = st.text_input("Ask something about your PDF:")
    if query:
        response = qa.run(query)
        st.markdown(f"**Answer:** {response}")

    # Clean temp file after use
    os.remove(pdf_path)