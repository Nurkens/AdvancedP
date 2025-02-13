# Knowledge Base Chat Assistant with RAG

This project is a Knowledge Base Chat Assistant application that combines a **Retrieval-Augmented Generation (RAG)** approach with an interactive chat interface. Users can query a knowledge base, perform Google searches, and receive contextual answers powered by LLMs.

---

## Features

- **User Authentication**: Secure login/signup with hashed passwords and JWT token management.
- **Retrieval-Augmented Generation (RAG)**:
  - Query documents stored in ChromaDB.
  - Perform Google searches for additional context if the knowledge base lacks relevant results.
- **Profanity Filtering**: Filters inappropriate content in both user inputs and usernames.
- **WordCloud Visualization**: Generates word clouds from text inputs.
- **Chat History**: Saves and loads user-specific chat history.
- **File Processing**:
  - Extract text from PDFs and plain text files.
- **Customizable Search Settings**:
  - Adjustable number of results from ChromaDB and Google Search.
- **Interactive Chat Interface**: Users can view questions, AI responses, and their sources directly in the interface.

---

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) for the UI.
- **Backend**:
  - [ChromaDB](https://github.com/chroma-core/chroma) for persistent vector storage.
  - [Sentence Transformers](https://www.sbert.net/) for text embeddings.
  - [LangChain](https://github.com/hwchase17/langchain) for prompt handling.
- **LLM**: Llama3 via Ollama API.
- **Storage**: SQLite for user credentials and JSON for chat history.
- **Others**: WordCloud, Matplotlib, Better-Profanity.

---

## Prerequisites

1. Python 3.9+
2. Install dependencies:

```bash
pip install -r requirements.txt


Set up a .env file with the following keys:


JWT_SECRET=<your_secret_key>
JWT_ALGORITHM=HS256
JWT_EXP_DELTA_SECONDS=3600
SERPAPI_KEY=<your_serpapi_key>



Ensure Llama3 is running locally via Ollama API on http://localhost:11434.


How to

Initialize the SQLite database for user authentication:

python -c "import sqlite3; conn = sqlite3.connect('users.db'); c = conn.cursor(); c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)); conn.commit(); conn.close()"


Start the Streamlit application:
streamlit run app.py


Usage
Sign up or log in:
Use the username and password fields to create an account or log in.
Ask questions:
Enter a query in the input box.
The assistant searches the knowledge base and web for relevant information.
Upload files:
Drag and drop PDF or text files to extract and query their content.
Generate word clouds:
Provide text input to create a visual representation.
```
