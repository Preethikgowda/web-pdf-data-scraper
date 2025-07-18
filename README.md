# web-pdf-data-scraper

# 📄 Web & PDF Data Scraper Assistant

A smart AI-powered document assistant that extracts and analyzes content from **PDF files** and **web URLs**, powered by **LangChain**, **Groq LLMs**, **FAISS**, and **Streamlit**.

## 🚀 Features

- 🌐 **Web Scraping**: Extracts clean content from any valid webpage.
- 📄 **PDF Analysis**: Upload and analyze multiple PDF documents with LLM-backed intelligence.
- 💬 **AI Chat Interface**: Ask questions and get accurate answers from scraped content.
- 📊 **Chat History & Deletion**: Manage your chat interactions and review previous queries.
- 🧠 **Embeddings with HuggingFace**: High-quality vector representation using `all-MiniLM-L6-v2`.
- 🎨 **Custom UI**: Styled with Tailwind-like elements for a professional feel.

---

## 🛠️ Tech Stack

| Tool          | Purpose                                      |
|---------------|----------------------------------------------|
| 🧠 LangChain   | Document processing & LLM chaining           |
| ⚡ Groq LLM    | Fast and accurate language model inference   |
| 🧬 HuggingFace | Text embeddings (`all-MiniLM-L6-v2`)         |
| 🗂️ FAISS       | Vector store for fast semantic search        |
| 📦 Streamlit   | Web UI for interaction                      |
| 🌐 WebBaseLoader / PyPDFLoader | Data loaders for web & PDF content |

---

## 📥 Installation

### 🔧 Prerequisites

- Python 3.8+
- `pip` package manager

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
🔐 Environment Variables
Create a .env file in the root directory and add your keys:

env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_key_if_applicable
🔒 Your .env is protected via .gitignore and won't be pushed.

🧪 Usage

streamlit run loader.py
Once launched, the app will allow you to:

Enter a URL or upload PDFs

Ask questions about the content

Get structured answers with document context



📚 Folder Structure

📁 Groq/
├── loader.py          # Main Streamlit app with full UI and logic
├── app.py             # Simple web-scraping demo
├── llama3.py          # PDF loader with Llama3-specific logic
├── requirements.txt   # Python dependencies
├── groq.ipynb         # Jupyter notebook for testing
└── .env               # API keys (excluded from Git)
📝 License
This project is licensed under the MIT License – feel free to use, modify, and share with attribution.

🤝 Contributions
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

🔗 Related Projects
LangChain

Streamlit

Groq API

FAISS

🙋‍♀️ Author
Developed by Preethi TK
preethikgowda26@gmail.com
