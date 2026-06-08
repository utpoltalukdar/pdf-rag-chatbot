import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

st.title("AI PDF Assistant")
st.write("Upload any PDF and ask questions about its contents.")

with st.sidebar:
    st.header("PDF RAG Chatbot")
    st.write("Upload a PDF and chat with it.")
    st.write("Powered by Gemini AI")

api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    st.success("Gemini API Key Loaded Successfully!")
else:
    st.error("API Key Not Found")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success(f"PDF uploaded successfully: {uploaded_file.name}")

    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vector_db.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

    question = st.text_input("Ask a question from this PDF")

if question:
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    Answer the question using only the PDF context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)