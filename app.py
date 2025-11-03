import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader, CSVLoader, TextLoader, PyPDFLoader
# from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
# from pinecone import Pinecone
from prompt import template


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_prompt_template()
        self.relative_path = 'templates'
        self.filename = 'doc.pdf'
        self.absolute_path = os.path.join(self.relative_path, self.filename)
        self.retriever = None
        self.initialize_retriever(self.absolute_path)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        # Initialize memory in Streamlit session state
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="history", input_key="question")

    def load_env_variables(self):
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        # self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        # self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    def setup_prompt_template(self):
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )

    def initialize_retriever(self, directory_path):
        loader = PyPDFLoader(directory_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        # Pinecone(api_key=self.pinecone_api_key, environment='gcp-starter')
        vectbd = Chroma.from_documents(
            docs, embeddings)
        self.retriever = vectbd.as_retriever()

    def finetune(self, file_path):
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type.")

        documents = loader.load_and_split() if hasattr(
            loader, 'load_and_split') else loader.load()
        self.process_documents(documents)
        os.remove(file_path)  # Remove the file after processing

    def process_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        # Pinecone(api_key=self.pinecone_api_key, environment='gcp-starter')
        vectbd = Chroma.from_documents(
            docs, embeddings)
        self.retriever = vectbd.as_retriever()

    def chat(self, user_input):
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={
                "verbose": False,
                "prompt": self.prompt_template,
                "memory": st.session_state.memory  # Use the memory from session state
            }
        )
        assistant_response = chain.invoke(user_input)
        response_text = assistant_response['result']

        return response_text


# Initialize the RAG Assistant
rag_assistant = RAGAssistant()

# Streamlit UI setup
st.set_page_config(page_title="Assistant", layout="wide")
st.title("RAG Assistant")

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for choosing between Chat and Fine-tuning
option = st.sidebar.selectbox("Choose an option", ("Chat", "Fine-tuning"))

if option == "Chat":
    st.header("Chat with your Docs")

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user prompt
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        response = rag_assistant.chat(prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

elif option == "Fine-tuning":
    st.header("Fine-tune RAG Assistant")
    uploaded_file = st.file_uploader(
        "Upload a file for fine-tuning", type=["txt", "pdf", "csv", "xlsx", "docx"])

    if uploaded_file is not None:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Fine-tuning in progress..."):
            rag_assistant.finetune(file_path)
        st.success(
            "Fine-tuning done successfully. You can now chat with the updated RAG Assistant.")
