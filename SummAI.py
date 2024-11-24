import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import whisper
from PyPDF2 import PdfReader
import docx
import tempfile
import os
from deep_translator import GoogleTranslator  # Import for translation

# Streamlit App Configuration
st.set_page_config(
    page_title="SummAI: Summarize and Chat",
    page_icon="❄️",
)
st.title("❄️ SummAI: Summarize and Chat")
st.subheader("Summarize URL, Audio, or Document and Chat with the Data")

# Sidebar for Groq API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

    # History Section in Sidebar
    st.subheader("📜 History")
    if "history" in st.session_state and st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            st.markdown(f"**{idx + 1}. {item['type']}**: {item['name']}")
            if st.button(f"View {item['type']} {idx + 1}", key=f"view_history_{idx}"):
                st.session_state.current_summary = item["summary"]
                st.session_state.current_qa_pairs = item["qa_pairs"]

# Dropdown and Input Box for Summarization Type
st.subheader("Choose the type of content to summarize")
content_type = st.selectbox("Select content type", ["URL", "Audio File", "Document File"])
input_content = None

if content_type == "URL":
    input_content = st.text_input("Enter the URL to summarize")
elif content_type == "Audio File":
    input_content = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
elif content_type == "Document File":
    input_content = st.file_uploader("Upload a document file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# LangChain LLM Setup (Gemma Model)
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Prompt Templates
map_prompt_template = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """
Write a concise summary of the following summaries:
"{text}"
CONCISE SUMMARY:
"""
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

qa_prompt_template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer the question thoroughly and accurately based on the context provided.
"""
qa_prompt = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"]
)

# Token limits
CHUNK_SIZE = 500  # Smaller chunks for better context retrieval
CHUNK_OVERLAP = 50

def extract_text_from_document(file):
    """Extract text from uploaded document files."""
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

def process_content(content, source):
    """Process content into chunks and create Document objects."""
    words = content.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i + CHUNK_SIZE + CHUNK_OVERLAP])
        chunks.append(Document(page_content=chunk, metadata={"source": source}))
    return chunks

def fetch_youtube_transcript(video_id):
    """Fetch and optionally translate a YouTube transcript."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join([t["text"] for t in transcript])
    except NoTranscriptFound:
        try:
            transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(['hi', 'es', 'fr', 'de'])
            translated_transcript = transcript.translate('en')
            return " ".join([t['text'] for t in translated_transcript.fetch()])
        except Exception:
            st.error("Transcripts are unavailable in translatable languages for this video.")
            return ""
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return ""

# Initialize session states
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_summary" not in st.session_state:
    st.session_state.current_summary = ""
if "current_qa_pairs" not in st.session_state:
    st.session_state.current_qa_pairs = []
if "history" not in st.session_state:
    st.session_state.history = []

# Embedding initialization
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Button for Processing Content
if st.button("Process Content"):
    if not groq_api_key.strip() or not input_content:
        st.error("Please provide valid input and Groq API Key.")
    else:
        try:
            with st.spinner("Processing content..."):
                docs = []
                source_name = ""

                # Process content based on selected type
                if content_type == "URL":
                    source_name = input_content
                    if "youtube.com" in input_content or "youtu.be" in input_content:
                        video_id = (
                            input_content.split("v=")[-1].split("&")[0]
                            if "youtube.com" in input_content
                            else input_content.split("/")[-1]
                        )
                        content = fetch_youtube_transcript(video_id)
                        if content:
                            docs.extend(process_content(content, input_content))
                    else:
                        try:
                            loader = UnstructuredURLLoader(urls=[input_content])
                            unstructured_docs = loader.load()
                            for doc in unstructured_docs:
                                docs.extend(process_content(doc.page_content, input_content))
                        except Exception as e:
                            st.error(f"Failed to load URL content: {e}")

                elif content_type == "Audio File":
                    source_name = input_content.name
                    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                        temp_file.write(input_content.read())
                        temp_file.flush()
                        model = whisper.load_model("base")
                        transcription = model.transcribe(temp_file.name)
                        audio_content = transcription["text"]
                        docs.extend(process_content(audio_content, "Uploaded Audio"))

                elif content_type == "Document File":
                    source_name = input_content.name
                    doc_content = extract_text_from_document(input_content)
                    docs.extend(process_content(doc_content, input_content.name))

                if not docs:
                    st.error("No content found. Please check the input and try again.")
                else:
                    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

                    summarize_chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        map_prompt=map_prompt,
                        combine_prompt=combine_prompt,
                        verbose=True
                    )
                    st.session_state.current_summary = summarize_chain.run(docs)
                    st.session_state.current_qa_pairs = []

                    # Add to history
                    st.session_state.history.append({
                        "type": content_type,
                        "name": source_name,
                        "summary": st.session_state.current_summary,
                        "qa_pairs": []
                    })
                    
                    st.success("Content processed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

# Display the current summary and Q&A
if st.session_state.current_summary:
    st.subheader("Summary")
    st.write(st.session_state.current_summary)

if st.session_state.vectorstore:
    st.subheader("Chat with the Data")
    user_question = st.text_input("Ask a question about the content:")
    
    if user_question:
        with st.spinner("Thinking..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                chain_type_kwargs={"prompt": qa_prompt},
                return_source_documents=True,
            )
            
            # Use `__call__` to get both the result and source documents
            response = qa_chain({"query": user_question})
            result = response["result"]  # Extract the answer
            source_documents = response["source_documents"]  # Extract source documents
            
            # Display the answer
            st.write("**Answer:**", result)
            
            # Optionally display source documents (if needed)
            with st.expander("Source Documents"):
                for doc in source_documents:
                    st.write(f"- {doc.page_content[:200]}...")  # Show snippet of the source
            
            # Save Q&A pairs in history
            qa_pair = {
                "question": user_question,
                "answer": result,
                "sources": [doc.metadata.get("source", "Unknown Source") for doc in source_documents]
            }
            st.session_state.current_qa_pairs.append(qa_pair)
            st.session_state.history[-1]["qa_pairs"].append(qa_pair)

# Sidebar History Section
with st.sidebar:
    st.subheader("History")
    if st.session_state.history:
        for idx, entry in enumerate(st.session_state.history):
            with st.expander(f"Session {idx + 1}"):
                # Display Summary
                st.write("**Summary:**", entry["summary"])
                
                # Display Q&A Pairs
                if "qa_pairs" in entry and entry["qa_pairs"]:
                    st.write("**Q&A Pairs:**")
                    for qidx, qa in enumerate(entry["qa_pairs"]):
                        st.write(f"**Q{qidx + 1}:** {qa['question']}")
                        st.write(f"**A{qidx + 1}:** {qa['answer']}")
                        if "sources" in qa and qa["sources"]:
                            st.write("**Sources:**")
                            for source in qa["sources"]:
                                st.write(f"- {source}")
    else:
        st.write("No history available.")