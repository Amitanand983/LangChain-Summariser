# 🦜 LangChain Summarizer App

This application leverages **LangChain** and **Groq's Gemma Model** to summarize content from YouTube videos, websites, audio files, or document files. It extracts relevant information and generates concise summaries, making it easy to analyze large amounts of text in seconds.

---

## 🚀 Features

- **Summarize YouTube Videos**: Extracts transcripts and generates a concise summary of the video content.  
- **Website Summarization**: Supports summarizing text from any web page.  
- **Audio File Transcription**: Transcribes and summarizes audio files (e.g., `.mp3`, `.wav`, `.m4a`).  
- **Document Summarization**: Handles `.pdf`, `.docx`, and `.txt` files for summarization.  

---

## 🛠️ Tech Stack

- **[LangChain](https://langchain-langchain.com)**: Provides LLM-based summarization capabilities.
- **[Groq Gemma Model](https://www.groq.com)**: A cutting-edge language model for natural language understanding.
- **Streamlit**: For a lightweight and interactive user interface.
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

3. **Upload Input or Provide URL**:
   - Enter a YouTube or Website URL.
   - Upload an audio file (`.mp3`, `.wav`, `.m4a`).
   - Upload a document (`.pdf`, `.docx`, `.txt`).

---

## 🖥️ Usage

1. **Start the app**: Open the app in your browser (default: `http://localhost:8501`).
2. **Select Input**:
   - Enter a YouTube URL for video summarization.
   - Paste a website URL to summarize webpage content.
   - Upload an audio file or document for transcription and summarization.
3. **Click "Summarize the Content"**: The app will process the input and display the summary.

---

## 🛑 Known Limitations

- **Token Limit**: Very large inputs (e.g., long videos or lengthy documents) may require chunking for processing.
- **YouTube Restrictions**: Videos with disabled transcripts cannot be processed.
- **Model Dependency**: Relies on Groq's language model for summarization, requiring an active API key.

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
```

This includes everything from cloning the repo, setting up the environment, running the app, and using it, along with features, contributions, and license information.
