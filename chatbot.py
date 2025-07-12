from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os

def setup_rag_chain(data_folder = "data"):
    # Load documents from the data folder
    document = []
    for file in os.listdir(data_folder):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_folder, file), encoding="utf-8")
            document.extend(loader.load())

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)

     # Create embeddings using a lightweight Hugging Face model
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

     # Store chunks in Chroma vector store
    vectorstore = Chroma.from_documents(texts, embeddings)

    # Initialize the language model (Hugging Face's flan-t5-base)
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={"max_length": 512}
    ) 

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )

    return qa_chain

def ask_question(qa_chain, query):
    # Run the query through the RAG chain
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]        