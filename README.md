# YouTube Q&A Bot

This app allows users to ask questions about a YouTube video. It fetches the transcript, chunks it, stores it in a vector database, and then uses an LLM to answer user questions based on the video content.

### [Demo](https://drive.google.com/file/d/1R0DSEk5j59MwWvTS3Wlu2QjVROOKH4DK/view?usp=sharing)

Built using:
- LangChain
- Ollama (Llama 3.1)
- FAISS (for vector DB)
- YouTube Transcript API
- Streamlit (for UI)

---

## Features

- Extracts and indexes transcript from any YouTube video (no audio download)
- Uses local LLM (Llama 3.1 via Ollama)
- Semantic search with FAISS
- Simple chat-style interface in Streamlit

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/youtube-qa-bot.git
cd youtube-qa-bot
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 4. Install and run Ollama

Install [Ollama](https://ollama.com/) and run the LLM:
```bash
ollama run llama3
```

### 5. Run the app
```bash
streamlit run main.py
```

Notes
	•	The video ID is hardcoded for now. You can update the video_id in create_vector_store() function.
	•	Works only for YouTube videos that have transcript available.
	•	You can swap in OpenAI or any other embedding/LLM easily.
