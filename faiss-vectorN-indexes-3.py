import os
import sys
import pdfplumber
import time  # For generating Unix timestamp
from datetime import datetime  # For getting the current date and time
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from port_checker import check_port, check_curl_response
import warnings


PORT = 11434
warnings.filterwarnings("ignore")

class Document:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}

def load_text_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(content))
        elif filename.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        documents.append(Document(content))
    return documents

def process_vector_data(directory_path, index_name):
    # Load text and PDF data from the specified directory
    text_documents = load_text_documents(directory_path)

    # Split the text data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(text_documents)

    # Create embeddings
    hugg_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create FAISS index
    faiss_vectorstore = FAISS.from_documents(documents=all_splits, embedding=hugg_embeddings)

    # Save the FAISS index to disk
    faiss_vectorstore.save_local(index_name)

    return faiss_vectorstore

def load_new_data(text_data, faiss_vectorstore, index_name):
    # Split the new text data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents([Document(text_data)])

    # Create embeddings
    hugg_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Update the FAISS index with the new data
    faiss_vectorstore.add_documents(documents=all_splits, embedding=hugg_embeddings)
    
    # Save the updated FAISS index to disk with a specific name
    faiss_vectorstore.save_local(index_name)
    
    return faiss_vectorstore

def check_port(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python faiss-llm-feedback.py <llm model>")
        sys.exit(1)

    llm_model = sys.argv[1]
    print(f"\nReceived LLM Model: {llm_model}")

    # Process vectorDataOne
    print("Processing vectorDataOne...")
    vectorDataOne_index = "vectorDataOne_index"
    faiss_vectorstore_one = process_vector_data("./vectorDataOne", vectorDataOne_index)

    # Process vectorDataTwo
    print("Processing vectorDataTwo...")
    vectorDataTwo_index = "vectorDataTwo_index"
    faiss_vectorstore_two = process_vector_data("./vectorDataTwo", vectorDataTwo_index)

    # Process vectorDataThree
    print("Processing vectorDataThree...")
    vectorDataThree_index = "vectorDataThree_index"
    faiss_vectorstore_three = process_vector_data("./vectorDataThree", vectorDataThree_index)

    # Merge all three FAISS indexes
    faiss_vectorstore_one.merge_from(faiss_vectorstore_two)
    faiss_vectorstore_one.merge_from(faiss_vectorstore_three)

    # Save the merged index to disk (optional)
    merged_index_name = "merged_vectorData_index"
    faiss_vectorstore_one.save_local(merged_index_name)

    # Convert the FAISS vector store into a retriever
    faiss_retriever = faiss_vectorstore_one.as_retriever(search_kwargs={"k": 5})

    # Create the QA prompt template
    template = """Use the following pieces of context to answer the question at the end.
    If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    # Initialize the LLM
    llm = Ollama(model=llm_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Check if the port is open initially
    if not check_port(PORT):
        print(f"\nPort {PORT} is not open. Please start the LLM engine.")
        sys.exit(1)

    print(f"\n\nCheck for story in folder ./text_files_directory about Remy.")
    print(f"\nPrompt suggestions:\nwhat is this story about?")

    # Interactive query input loop
    while True:
        if not check_port(PORT):
            print(f"\nPort {PORT} is not open. Exiting...")
            print(f"\nPlease start the LLM engine on port {PORT}.")
            sys.exit(1)

        query = input("\n\nEnter your query (type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        # Process the query
        result = qa_chain({"query": query})

        # Generate filename based on Unix timestamp
        timestamp = int(time.time())
        filename = f"text_files_directory/result_{timestamp}.txt"
        
        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"Date and Time: {current_time}\n\n{str(result)}")

        # Automatically load the result content into all FAISS indexes for the next query
        faiss_vectorstore_one = load_new_data(str(result), faiss_vectorstore_one, merged_index_name)

        # Update the retriever with the new FAISS index
        faiss_retriever = faiss_vectorstore_one.as_retriever(search_kwargs={"k": 5})

        # Update the QA chain with the new retriever
        qa_chain.retriever = faiss_retriever
        