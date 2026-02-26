## RAG Q&A Conversation With PDF Including Chat History (LangChain 1.2.x)

import streamlit as st
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Conversational RAG With PDF uploads and Chat History")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = []

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        documents = []

        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # -----------------------------
        # Split Documents
        # -----------------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # -----------------------------
        # Vector Store
        # -----------------------------
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        # -----------------------------
        # Context Formatter
        # -----------------------------
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # -----------------------------
        # Prompt
        # -----------------------------
        system_prompt = """
        You are an assistant for question-answering tasks.
        Use the retrieved context to answer the question.
        If you don't know the answer, say you don't know.
        Keep the answer concise (max 3 sentences).

        Context:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ]
        )

        # -----------------------------
        # RAG Chain (LCEL Style)
        # -----------------------------
        rag_chain = (
            {
                "context": lambda x: format_docs(
                    retriever.invoke(x["question"])
                ),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # -----------------------------
        # User Input
        # -----------------------------
        user_input = st.text_input("Your Question:")

        if user_input:

            chat_history = st.session_state.store[session_id]

            response = rag_chain.invoke(
                {
                    "question": user_input,
                    "chat_history": chat_history,
                }
            )

            # Save chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))

            st.write("Assistant:", response)

            with st.expander("Chat History"):
                st.write(chat_history)

else:
    st.warning("Please enter the Groq API Key")