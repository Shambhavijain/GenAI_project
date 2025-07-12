import streamlit as st
from retriever import setup_retriever, retrieve
from generator import generate_answer

st.set_page_config(page_title="Loan Approval Q&A Chatbot", layout="centered")
st.title("Loan Approval Q&A Chatbot 🤖")

# Cache and load retriever setup
@st.cache_resource
def init():
    data_path = "data/Training Dataset.csv"
    texts, model, index = setup_retriever(data_path)
    return texts, model, index

texts, model, index = init()

# User input
query = st.text_input("Ask a question about loan approval:")

# Handle user query
if query:
    with st.spinner("Thinking..."):
        top_docs = retrieve(query, model, index, texts)
        answer = generate_answer(query, top_docs)
        st.subheader("Answer:")
        st.write(answer)
