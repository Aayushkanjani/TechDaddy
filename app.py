import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='all-minilm')
    st.session_state.loader = PyPDFLoader("TheIIITAllahabadHandbook.pdf")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:5])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("TechDaddy")
st.sidebar.header("Configuration")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Load the SentenceTransformer model for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(query, documents):
    query_embedding = semantic_model.encode([query])
    doc_embeddings = semantic_model.encode([doc.page_content for doc in documents])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    return similarities

prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    context_docs = response["context"]

    # Compute semantic similarity between the input query and retrieved documents
    similarities = compute_semantic_similarity(prompt, context_docs)

    # Combine documents with their similarity scores and sort by similarity
    sorted_docs = sorted(zip(context_docs, similarities), key=lambda x: x[1], reverse=True)

    # Display response and the most relevant documents
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, (doc, sim) in enumerate(sorted_docs):
            st.write(f"Document {i+1} (Similarity: {sim:.2f}):")
            st.write(doc.page_content)
            st.write("--------------------------------")

    print("Response time:", time.process_time() - start)
