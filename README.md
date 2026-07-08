# 🎥 YouTube AI Video RAG Assistant

A **Retrieval-Augmented Generation (RAG)** application that allows users to analyze YouTube videos using AI. Simply paste a YouTube video URL, and the application extracts the transcript, generates an AI-powered summary, and enables users to ask natural language questions based solely on the video's content.

The project leverages **LangChain**, **FAISS**, **Hugging Face Embeddings**, **Groq Llama 3.1**, and **Streamlit** to build a fast, accurate, and interactive AI-powered video assistant.

---

## 🚀 Features

- Analyze any YouTube video using its URL
- Automatically extract video transcripts
- Generate:
  - Executive Summary
  - Key Points
  - Objectives
  - Important Concepts
  - Conclusion
- Ask questions about the video in natural language
- Retrieval-Augmented Generation (RAG) for transcript-grounded answers
- Semantic search using FAISS Vector Store
- Hugging Face Embeddings for efficient document retrieval
- Fast inference using Groq Llama 3.1
- Interactive Streamlit interface

---

## 🏗️ Architecture

```
                 User
                   │
                   ▼
          YouTube Video URL
                   │
                   ▼
          Extract Video ID
                   │
                   ▼
    YouTube Transcript API
                   │
                   ▼
      Recursive Text Splitter
                   │
                   ▼
    Hugging Face Embeddings
                   │
                   ▼
         FAISS Vector Store
                   │
                   ▼
          Semantic Retrieval
                   │
                   ▼
          Groq Llama 3.1
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  Video Summary         Question Answering
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Groq (Llama 3.1 8B Instant)
- FAISS
- Hugging Face Embeddings
- YouTube Transcript API

---

## 📂 Project Structure

```
youtube-ai-video-rag-assistant
│
├── app.py                # Streamlit frontend
├── main.py               # RAG pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Tamanna-Aggarwal-2004/youtube-ai-video-rag-assistant.git

cd youtube-ai-video-rag-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📖 How It Works

1. User enters a YouTube video URL.
2. The application extracts the video ID.
3. Retrieves the transcript using the YouTube Transcript API.
4. Splits the transcript into chunks using LangChain.
5. Generates embeddings using Hugging Face Embeddings.
6. Stores embeddings in a FAISS Vector Store.
7. Retrieves the most relevant transcript chunks for each query.
8. Uses Groq Llama 3.1 to:
   - Generate structured summaries
   - Answer questions using only the retrieved transcript context

---

## 📌 Future Improvements

- Support multilingual transcripts
- Timestamp-based answers
- Video chapter generation
- Hybrid search with reranking
- Conversation history
- PDF export for summaries
- Support additional LLM providers

---

## 👩‍💻 Author

**Tamanna Aggarwal**

- GitHub: https://github.com/Tamanna-Aggarwal-2004
- LinkedIn: https://www.linkedin.com/in/tamanna-aggarwal-2004/

---

⭐ If you found this project useful, consider giving it a **Star**!
