import streamlit as st
import hashlib
import sqlite3
import os
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import OllamaLLM
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from datetime import datetime
import pandas as pd
from better_profanity import profanity
import re
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# --- Constants and Initialization ---
llm_model = "llama3"
chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
collection_name = "knowledge_base"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize ChromaDB
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(name=collection_name, metadata={"description": "Knowledge base for RAG"})

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Добавляем новые константы
SERPAPI_KEY = "227e6513442cfaa35c62f17d43e8f0de50c3450af033c0ed8f9e7bee4de393c2"
MAX_SEARCH_RESULTS = 5
CHAT_HISTORY_FILE = "chat_history.json"

# Инициализация фильтра нецензурных слов
profanity.load_censor_words()

# --- Core Functions ---
def generate_embeddings(documents):
    return embedding_model.encode(documents)

def save_embeddings(documents, ids):
    embeddings = generate_embeddings(documents)
    collection.add(documents=documents, ids=ids, embeddings=embeddings)

def query_chromadb(query_text, n_results=1):
    query_embedding = generate_embeddings([query_text])
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results["documents"][0] if results["documents"] else "No relevant documents found."

def google_search(query, num_results=MAX_SEARCH_RESULTS):
    if not SERPAPI_KEY:
        st.error("SerpAPI key not configured!")
        return ""
    
    url = f"https://serpapi.com/search.json?q={query}&num={num_results}&api_key={SERPAPI_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json().get("organic_results", [])
        if results:
            return "\n\n".join(f"Source: {item.get('link', 'Unknown')}\n{item.get('snippet', '')}" 
                             for item in results)
        return ""
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing SerpAPI: {str(e)}")
        return ""
    except json.JSONDecodeError:
        st.error("Error parsing SerpAPI response")
        return ""

def wrap_text(text, max_length=80):
    return "\n".join([text[i:i + max_length] for i in range(0, len(text), max_length)])

def extract_text_from_file(file):
    if file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
        os.unlink(temp_file.name)
        return [doc.page_content for doc in documents]
    elif file.type == "text/plain":
        return [file.read().decode("utf-8")]
    else:
        st.error("Unsupported file type.")
        return []

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def process_query():
    if 'messages' not in st.session_state:
        st.session_state.messages = load_chat_history(st.session_state.username)

    # Настройки поиска
    with st.sidebar.expander("Search Settings"):
        num_chroma_results = st.slider("Number of ChromaDB results", 1, 10, 3)
        num_google_results = st.slider("Number of Google results", 1, 10, 5)
        enable_profanity_filter = st.checkbox("Enable Profanity Filter", value=True)

    # История чата
    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "source" in message:
                    st.caption(f"Source: {message['source']}")

    user_input = st.text_input("Ask a question:")

    if user_input:
        # Применяем фильтр нецензурных слов, если он включен
        if enable_profanity_filter:
            filtered_input = filter_profanity(user_input)
            if filtered_input != user_input:
                st.warning("Your message contained inappropriate content and was filtered.")
            user_input = filtered_input

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Searching knowledge base and web..."):
            # Поиск в ChromaDB
            chroma_context = query_chromadb(user_input, n_results=num_chroma_results)
            
            # Если в ChromaDB ничего не найдено, ищем в Google
            if chroma_context == "No relevant documents found.":
                context = google_search(user_input, num_results=num_google_results)
                source = "web search"
            else:
                context = chroma_context
                source = "knowledge base"

            prompt = f"""
            Question: {user_input}
            
            Context from {source}:
            {context}
            
            Please provide a comprehensive answer based on the context above.
            If using web search results, cite the sources.
            
            Answer:
            """

            try:
                llm = OllamaLLM(model=llm_model, base_url="http://localhost:11434")
                response = llm.invoke(prompt)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                })

                with st.chat_message("assistant"):
                    st.write(response)
                    st.caption(f"Source: {source}")

                # Сохраняем историю чата
                save_chat_history(st.session_state.username, st.session_state.messages)

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# --- Authentication Functions ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0] == make_hash(password)
    return False

def add_user(username, password):
    # Проверяем имя пользователя на нецензурные слова
    if profanity.contains_profanity(username):
        st.error("Username contains inappropriate content.")
        return False
        
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users VALUES (?,?)', (username, make_hash(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''

def save_chat_history(username, messages):
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = {}
        
        history[username] = {
            'messages': messages,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

def load_chat_history(username):
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                return history.get(username, {}).get('messages', [])
        return []
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return []

def create_wordcloud(text):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"Error creating wordcloud: {str(e)}")
        return None

def filter_profanity(text):
    try:
        # Базовая фильтрация с помощью better-profanity
        filtered_text = profanity.censor(text)
        
        # Дополнительная фильтрация для русских слов (опционально)
        russian_profanity = [
            'блять', 'сука', 'хуй', 'пизда', 'ебать',
            # Добавьте другие слова по необходимости
        ]
        
        for word in russian_profanity:
            pattern = re.compile(word, re.IGNORECASE)
            filtered_text = pattern.sub('****', filtered_text)
            
        return filtered_text
    except Exception as e:
        st.error(f"Error in profanity filter: {str(e)}")
        return text

def main():
    init_db()
    init_session_state()

    st.set_page_config(page_title="Knowledge Navigator", layout="wide")

    if not st.session_state.logged_in:
        st.title("Welcome to Knowledge Navigator")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.header("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if check_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            st.header("Register")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            
            if st.button("Register"):
                if reg_password != reg_password_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif add_user(reg_username, reg_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

    else:
        st.title(f"Welcome back, {st.session_state.username}!")
        
        # Боковая панель
        with st.sidebar:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ''
                st.rerun()
            
            st.divider()
            
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                save_chat_history(st.session_state.username, [])
                st.rerun()

        # Основные вкладки
        tab1, tab2, tab3 = st.tabs(["Chat", "Documents", "Analytics"])
        
        with tab1:
            with st.expander("💬 Chat with the AI", expanded=True):
                process_query()
        
        with tab2:
            st.subheader("📚 Manage Documents")
            uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
            new_doc = st.text_area("Alternatively, enter document text manually:")

            if uploaded_file:
                document_text = extract_text_from_file(uploaded_file)
                if document_text:
                    # Применяем фильтр к тексту документа
                    filtered_text = [filter_profanity(content) for content in document_text]
                    wrapped_text = [wrap_text(content) for content in filtered_text]
                    for i, content in enumerate(wrapped_text):
                        new_id = f"doc_{collection.count() + 1}_{i}"
                        save_embeddings([content], [new_id])
                    st.success("Document added successfully!")
                    
                    # Создаем облако слов для загруженного документа
                    st.subheader("📊 Document Visualization")
                    combined_text = " ".join(wrapped_text)
                    wordcloud_fig = create_wordcloud(combined_text)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
            
            elif new_doc.strip():
                # Применяем фильтр к введенному тексту
                filtered_doc = filter_profanity(new_doc)
                if filtered_doc != new_doc:
                    st.warning("The document contained inappropriate content and was filtered.")
                new_id = f"doc_{collection.count() + 1}"
                save_embeddings([filtered_doc], [new_id])
                st.success("Document added successfully!")
        
        with tab3:
            st.subheader("📊 Analytics")
            if st.session_state.messages:
                # Анализ истории чата
                df = pd.DataFrame([
                    {
                        'timestamp': msg.get('timestamp', ''),
                        'role': msg['role'],
                        'source': msg.get('source', ''),
                        'content_length': len(msg['content'])
                    }
                    for msg in st.session_state.messages
                ])
                
                if not df.empty:
                    st.write("Chat Statistics:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Messages", len(df))
                        st.metric("User Messages", len(df[df['role'] == 'user']))
                    with col2:
                        st.metric("AI Responses", len(df[df['role'] == 'assistant']))
                        sources = df[df['source'].notna()]['source'].value_counts()
                        st.write("Information Sources:", sources.to_dict())

if __name__ == "__main__":
    main()
