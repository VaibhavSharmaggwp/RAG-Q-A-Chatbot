import streamlit as st
from chatbot import setup_rag_chain, ask_question
import os

st.title("RAG Q&A Chatbot")
st.write("Upload text files or use existing files in the data folder to ask questions about their content!")

# Create data folder if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# File uploader
uploaded_files = st.file_uploader("Upload additional text files", type=["txt"], accept_multiple_files=True)

# Save uploaded files to data folder
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully!")

# Initialize RAG chain if data folder has files
if os.listdir("data"):
    with st.spinner("Processing documents (this may take a few minutes for large files like long_text.txt)..."):
        qa_chain = setup_rag_chain()
else:
    qa_chain = None
    st.warning("Please upload at least one text file or ensure the data folder contains text files.")

# Chat interface
st.subheader("Ask a Question")
query = st.text_input("Enter your question:")
if st.button("Submit") and query and qa_chain:
    with st.spinner("Generating answer..."):
        answer, sources = ask_question(qa_chain, query)
        st.write("**Answer**:")
        st.write(answer)
        st.write("**Sources**:")
        for i, doc in enumerate(sources):
            st.write(f"Document {i+1}: {doc.page_content[:200]}...")
else:
    if query and not qa_chain:
        st.error("No documents found. Please upload a text file or check the data folder.")