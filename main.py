from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

load_dotenv()

llm = OllamaLLM(model="llama3.1")
embedding_model = OllamaEmbeddings(model="llama3.1")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    Just answer the exact question, don't explain.
    <context> {context} </context>
    Question: {input}"""
)

output_parser = StrOutputParser()

# function to create and store vector once
def create_vector_store():
    video_id = "opi1s_5Dm-c"  # Just the video ID, not full URL
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    full_text = " ".join([entry["text"] for entry in transcript])

    doc = Document(page_content=full_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents([doc])

    store = FAISS.from_documents(docs, embedding_model)
    return store.as_retriever()

# Initialize vector store and qa history in session state
if 'retrieval' not in st.session_state:
    st.session_state['retrieval'] = create_vector_store()
    st.session_state['qa_history'] = []

stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(st.session_state['retrieval'], stuff_documents_chain)

st.title("YouTube Q&A Bot")

user_input = st.text_input("You: ")

if user_input:
    if user_input.lower() == "exit":
        st.write("Chat ended.")
    else:
        st.session_state['qa_history'].append(f"You: {user_input}")
        response = retrieval_chain.invoke({'input': user_input})
        st.session_state['qa_history'].append(f"Bot: {response['answer']}")
        
        # Display the entire qa history
        for message in st.session_state['qa_history']:
            st.write(message)