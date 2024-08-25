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
        self.metadata = {}  # Adding metadata attribute

def load_new_data(text_data, faiss_vectorstore):
    # Split the new text data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents([Document(text_data)])

    # Create embeddings
    hugg_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Update the FAISS index with the new data
    faiss_vectorstore.add_documents(documents=all_splits, embedding=hugg_embeddings)
    
    # Save the updated FAISS index to disk
    faiss_vectorstore.save_local("vectorDataOne_index")
    
    return faiss_vectorstore

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


if len(sys.argv) != 2:
    print("Usage: python faiss-llm-feedback.py <llm model>")
    sys.exit(1)
    
# Retrieve the variable from command-line arguments
llm_model = sys.argv[1]
    
# Print the variable or use it in your logic
print(f"\nReceived LLM Model: {llm_model}")


# Load text and PDF data from directory
text_documents = load_text_documents("./vectorDataOne")  # Update this path to your directory

# Split the text data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(text_documents)

# Create embeddings
hugg_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create FAISS index
faiss_vectorstore = FAISS.from_documents(documents=all_splits, embedding=hugg_embeddings)

# Save the FAISS index to disk
faiss_vectorstore.save_local("vectorDataOne_index")

# Convert the FAISS vector store into a retriever
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

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
    sys.exit(1)  # Exit the program with a non-zero status code


print(f"\n\nCheck for story in folder ./text_files_directory about Remy.")
print(f"\nPrompt suggestions:\nwhat is this story about?\nis there a dragon?\ntell me more about Remy.\ncreate another adventure under the ocean for Remy.")

# Interactive query input loop
while True:

    # Check if the port is open
    if not check_port(PORT):
        print(f"\nPort {PORT} is not open. Exiting...")
        print(f"\nPlease start the LLM engine on port {PORT}.")
        sys.exit(1)  # Exit the program with a non-zero status code


    query = input("\n\nEnter your query (type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Exiting...")
        break
    
    # Process the query
    result = qa_chain({"query": query})
    
    ##docs = retriever.get_relevant_documents(query)
    ##pretty_print_docs(docs)

    # Generate filename based on Unix timestamp
    timestamp = int(time.time())
    filename = f"text_files_directory/result_{timestamp}.txt"
    
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save result to text file with date and time appended
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Date and Time: {current_time}\n\n{str(result)}")
    
    ##print(f"Result saved to {filename}")
    
    # Automatically load the result content into the FAISS index for the next query
    faiss_vectorstore = load_new_data(str(result), faiss_vectorstore)

    # Update the retriever with the new FAISS index
    qa_chain.retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

