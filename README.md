# 🩺 Medical Chatbot using LangChain, Pinecone, HuggingFace, OpenAI & Flask

A **Retrieval-Augmented Generation (RAG)** chatbot that answers medical questions using text extracted from PDF documents.
The system uses **LangChain** for orchestration, **HuggingFace** for embeddings, **Pinecone** for vector storage, **OpenAI GPT-4o** for generating responses, and **Flask** for web deployment.

---

## 🩺 Overview

This project enables a **medical assistant chatbot** that:

* Extracts text from **medical PDFs**.
* Converts text into **semantic embeddings** using a **HuggingFace model**.
* Stores and retrieves text chunks from **Pinecone**.
* Generates concise, context-aware answers using **GPT-4o**.
* Serves responses through a **Flask web interface**.

---

## 🔄 Project Workflow

```
PDFs → Text Extraction → Text Splitting → Embedding → Pinecone Storage → User Query → Context Retrieval → GPT-4o Response → Flask UI
```

---

## 🧩 Detailed Code Explanation

---

## **1️⃣ helper.py**

This file handles **data ingestion and preprocessing** — extracting text from PDFs, cleaning it, splitting it into manageable chunks, and embedding them.

---

### 📄 Code

```python
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
```

These imports bring in:

* **`PyPDFLoader`** – reads and extracts text from each PDF page.
* **`DirectoryLoader`** – loads all PDFs in a folder.
* **`RecursiveCharacterTextSplitter`** – splits text into smaller overlapping chunks for embedding.
* **`HuggingFaceEmbeddings`** – generates vector embeddings.
* **`Document`** – a LangChain class representing text + metadata.

---

### 🧱 Function 1: Load PDFs

```python
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
```

**Explanation:**

* `DirectoryLoader` scans the given folder for all `.pdf` files.
* Each file is loaded using `PyPDFLoader`, which extracts page-level text.
* Returns a list of `Document` objects, each containing:

  ```python
  Document(page_content="Page text...", metadata={"source": "path/to/file.pdf"})
  ```

✅ **Purpose:** Extract all raw text data from your medical PDFs.

---

### 🧹 Function 2: Filter Minimal Metadata

```python
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs
```

**Explanation:**

* Loops through all `Document` objects.
* Keeps only essential metadata (the file name or source path).
* Removes page numbers, offsets, and other unnecessary info.

✅ **Purpose:** Keep the Pinecone index lightweight and clean.

---

### ✂️ Function 3: Split Text into Chunks

```python
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
```

**Explanation:**

* Splits long text into 500-character chunks, overlapping by 20 characters.
* Overlaps preserve sentence continuity between chunks.

✅ **Purpose:** Makes text suitable for embedding models (short, coherent, semantically meaningful).

---

### 🧠 Function 4: Load Embedding Model

```python
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
```

**Explanation:**

* Loads a **384-dimensional** embedding model from Hugging Face.
* Converts text into numerical vectors (representing meaning).

✅ **Purpose:** Generate embeddings that will be stored in Pinecone.

---

## **2️⃣ prompt.py**

This file defines the **system-level behavior** of your chatbot.

---

### 💬 Code

```python
system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
```

**Explanation:**

* **Defines the chatbot’s persona** as a “Medical Assistant.”
* Instructs the LLM to:

  * Use retrieved context (from Pinecone).
  * Avoid hallucination.
  * Keep answers short (max 3 sentences).
  * Say “I don’t know” if unsure.
* `{context}` will later be filled dynamically with retrieved text.

✅ **Purpose:** Controls the tone, factuality, and behavior of the chatbot.

---

## **3️⃣ store_index.py**

This script builds and uploads embeddings to Pinecone.
It’s **run once** before starting the chatbot.

---

### ⚙️ Code

```python
from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
```

Loads environment variables for API authentication.

---

### Step 1: Load API Keys

```python
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```

---

### Step 2: Extract and Preprocess Data

```python
extracted_data = load_pdf_file("./data")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)
```

1. Load all PDFs.
2. Clean metadata.
3. Split text into small chunks.

---

### Step 3: Create Embeddings and Connect to Pinecone

```python
embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)
```

---

### Step 4: Create Pinecone Index

```python
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
```

Creates a **vector index** for storing embeddings (only once).

---

### Step 5: Upload Embeddings

```python
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
```

✅ **Purpose:**
Pushes all chunks + embeddings to Pinecone so they can be retrieved later.

---

## **4️⃣ app.py**

This is the **main Flask application** that ties everything together — the UI, retriever, and LLM.

---

### 🔑 Step 1: Imports and Setup

```python
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()
```

Loads all necessary modules and initializes Flask.

---

### 🔐 Step 2: API Keys and Embeddings

```python
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
```

---

### 🪣 Step 3: Connect to Pinecone

```python
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
```

* Connects to the existing Pinecone index.
* Retrieves top 3 matching chunks for each question.

---

### 🤖 Step 4: LLM Setup & Prompt

```python
chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

Combines everything:

* Retrieves context → injects into `system_prompt` → sends to GPT-4o.

✅ **This is your Retrieval-Augmented Generation chain.**

---

### 💬 Step 5: Flask Routes

```python
@app.route("/")
def index():
    return render_template('chat.html')
```

Renders the chat interface.

```python
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])
```

Processes user queries and returns the LLM’s final answer.

---

### ▶️ Step 6: Run the App

```python
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
```

Runs the Flask server locally on port 8080.

---

## **5️⃣ setup.py**

Defines how your project can be installed as a Python package.

```python
from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Balaji S",
    author_email="balaji4991512@gmail.com",
    packages=find_packages(),
    install_requires=[]
)
```

✅ **Purpose:** Enables `pip install -e .` for easy local development.

---

## **6️⃣ template.sh**

Creates initial folder and file structure for the project.

```bash
# Creating directories
mkdir -p src
mkdir -p research

# Creating core files
touch src/__init__.py
touch src/helper.py
touch src/prompt.py
touch app.py
touch requirements.txt
```

✅ **Purpose:** Bootstrap script for setting up a clean project structure.

---

## **Frontend Files**

* **`templates/chat.html`** → Contains chat interface markup.
* **`static/style.css`** → Defines visual styling for the chat UI.

Example CSS snippet:

```css
body {
  background-color: #f4f8fb;
  font-family: Arial, sans-serif;
}
```

---

## ⚙️ How to Run

```bash
# Step 1: Create a virtual environment
conda create -n medibot python=3.10 -y
conda activate medibot

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Add API keys
echo "PINECONE_API_KEY=your_key" >> .env
echo "OPENAI_API_KEY=your_key" >> .env

# Step 4: Build the Pinecone index
python store_index.py

# Step 5: Start the chatbot
python app.py
```

Access at [http://localhost:8080](http://localhost:8080)

---

## 🔄 Summary of Flow

```
1. Load PDFs → 2. Split Text → 3. Generate Embeddings
→ 4. Store in Pinecone → 5. Retrieve Context
→ 6. Generate Answer (GPT-4o) → 7. Display in Flask UI
```
