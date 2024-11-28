# ❄️ SummAI – Multimodal Multitask Summarization and Q&A Tool

This updated application combines the power of **LangChain** and **Groq's Gemma Model** to summarize content from YouTube videos, websites, audio files, or document files and enables an interactive chat interface with the summarized data. This allows users not only to analyze large content but also to engage with it conversationally.

---

## 🚀 Features

- **Summarize YouTube Videos**: Extracts transcripts and generates concise summaries of video content.  
- **Website Summarization**: Summarizes text from any web page.  
- **Audio File Transcription and Summarization**: Transcribes and summarizes audio files (e.g., `.mp3`, `.wav`, `.m4a`).  
- **Document Summarization**: Handles `.pdf`, `.docx`, and `.txt` files for summarization.  
- **Chat with Summarized Data**: Enables users to ask questions and interact with the summarized content using a chatbot interface.

---

## 🛠️ Tech Stack

- **[LangChain](https://langchain-langchain.com)**: Provides LLM-based summarization and retrieval capabilities.
- **[Groq Gemma Model](https://www.groq.com)**: A state-of-the-art language model for natural language processing.
- **Streamlit**: For a dynamic and user-friendly web interface.
- **FAISS**: For efficient vector-based retrieval in the chatbot.
- **Whisper by OpenAI**: Transcribes audio files into text.

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- A [Groq API Key](https://www.groq.com) (required for the app)

### Clone the Repository
```bash
git clone https://github.com/amitanand983/langchain-summarizer.git
cd langchain-summarizer
```

### Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Setup

1. **Add Your Groq API Key**:
   - Go to the **sidebar** in the app.
   - Enter your **Groq API Key**.  

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Provide Input**:
   - Enter a YouTube or Website URL.
   - Upload an audio file (`.mp3`, `.wav`, `.m4a`).
   - Upload a document (`.pdf`, `.docx`, `.txt`).

---

## 🖥️ Usage

1. **Start the app**: Open the app in your browser (default: `http://localhost:8501`).
2. **Input Content**:
   - Enter a YouTube URL for video summarization.
   - Paste a website URL to summarize webpage content.
   - Upload an audio file or document for transcription and summarization.
3. **Summarize the Content**: Click "Summarize the Content" to generate a summary of the input. The app processes the input and displays a concise summary.
4. **Chat with the Data**:
   - Use the chat interface to ask questions about the summarized content.
   - View responses with relevant sources displayed for transparency.

---

## 🛑 Known Limitations

- **Token Limit**: Very large inputs (e.g., lengthy videos or documents) are chunked for processing, which may impact context consistency in summaries.
- **YouTube Restrictions**: Videos with disabled transcripts cannot be processed.
- **Model Dependency**: Requires an active Groq API Key for the Gemma model.

---

## 🤝 Contributions

Contributions are welcome! Feel free to open issues or submit pull requests for new features, bug fixes, or improvements.

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🧑‍💻 Author

Developed by [Amit Anand](https://github.com/amitanand983).

Feel free to reach out for suggestions or feedback!
