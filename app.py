import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
import whisper
from PyPDF2 import PdfReader
import docx
import tempfile

# Streamlit App Configuration
st.set_page_config(
    page_title="LangChain: Summarize Text From YT, Website, Audio, or Document",
    page_icon="🦜",
)
st.title("🦜 LangChain: Summarize Text From YT, Website, Audio, or Document")
st.subheader("Summarize URL, Audio, or Document")

# Sidebar for Groq API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Inputs
generic_url = st.text_input("URL", label_visibility="collapsed")
audio_file = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
document_file = st.file_uploader("Upload a document file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# LangChain LLM Setup (Gemma Model)
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Summarize the following content concisely:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Token limits
MAX_CHUNK_TOKENS = 1000  # Max tokens for each chunk, adjusted for safety
TOKEN_RESERVE = 1000  # Reserve tokens for metadata and prompts

def split_into_chunks(text, max_tokens):
    """Split text into chunks that fit the token limit."""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

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

def summarize_chunks(llm, chunks):
    """Summarize each chunk and combine results."""
    summaries = []
    summarize_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    for chunk in chunks:
        doc = Document(page_content=chunk, metadata={})
        summary = summarize_chain.run([doc])
        summaries.append(summary)
    return " ".join(summaries)

# Button for Summarization
if st.button("Summarize the Content"):
    if not groq_api_key.strip() or (not generic_url.strip() and not audio_file and not document_file):
        st.error("Please provide valid input and Groq API Key.")
    else:
        try:
            with st.spinner("Processing..."):
                docs = []

                # Process YouTube or Website URL
                if generic_url:
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        try:
                            video_id = (
                                generic_url.split("v=")[-1].split("&")[0]
                                if "youtube.com" in generic_url
                                else generic_url.split("/")[-1]
                            )
                            transcript = YouTubeTranscriptApi.get_transcript(video_id)
                            content = " ".join([t["text"] for t in transcript])
                            docs.append(Document(page_content=content, metadata={"source": generic_url}))
                        except TranscriptsDisabled:
                            st.error("Transcripts are disabled for this video. Please try another video.")
                        except Exception as e:
                            st.error(f"Failed to fetch YouTube content: {e}")
                    else:
                        loader = UnstructuredURLLoader(urls=[generic_url])
                        unstructured_docs = loader.load()
                        for doc in unstructured_docs:
                            docs.append(doc)

                # Process Audio File
                if audio_file:
                    model = whisper.load_model("base")
                    transcription = model.transcribe(audio_file.name)
                    audio_content = transcription["text"]
                    docs.append(Document(page_content=audio_content, metadata={"source": "Uploaded Audio"}))

                # Process Document File
                if document_file:
                    doc_content = extract_text_from_document(document_file)
                    docs.append(Document(page_content=doc_content, metadata={"source": document_file.name}))

                # Handle Empty or Invalid Data
                if not docs:
                    st.error("No content found. Please check the input and try again.")
                else:
                    # Split and Summarize
                    all_summaries = []
                    for doc in docs:
                        chunks = list(split_into_chunks(doc.page_content, MAX_CHUNK_TOKENS))
                        summary = summarize_chunks(llm, chunks)
                        all_summaries.append(summary)

                    final_summary = "\n\n".join(all_summaries)
                    st.success(final_summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)


# working version of youtube, website and audio file

# # app.py

# import validators
# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.chains.summarize import load_summarize_chain
# from langchain.schema import Document
# from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api._errors import TranscriptsDisabled
# import whisper  # Whisper for audio transcription
# import tempfile  # To handle temporary file storage

# # Streamlit App Configuration
# st.set_page_config(
#     page_title="LangChain: Summarize Text From YT, Website, or Audio",
#     page_icon="🦜",
# )
# st.title("🦜 LangChain: Summarize Text From YT, Website, or Audio")
# st.subheader('Summarize URL or Audio')

# # Sidebar for Groq API Key Input
# with st.sidebar:
#     groq_api_key = st.text_input("Groq API Key", value="", type="password")

# # URL Input Field
# generic_url = st.text_input("URL", label_visibility="collapsed")

# # Audio File Uploader
# audio_file = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

# # LangChain LLM Setup (Gemma Model)
# llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# # Prompt Template
# prompt_template = """
# Provide a summary of the following content in 300 words:
# Content:{text}
# """
# prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# # Token limit
# MAX_TOKENS = 15000  # The maximum tokens allowed by the model
# TOKEN_RESERVE = 1000  # Reserve tokens for system prompts and metadata

# def truncate_content(content, max_tokens):
#     """Truncate content to fit within the token limit."""
#     tokens = content.split()
#     if len(tokens) > max_tokens:
#         truncated = " ".join(tokens[:max_tokens])
#         return truncated, len(tokens)
#     return content, len(tokens)

# # Button for Summarization
# if st.button("Summarize the Content from YT, Website, or Audio"):
#     # Validate Inputs
#     if not groq_api_key.strip() or (not generic_url.strip() and not audio_file):
#         st.error("Please provide the information to get started.")
#     elif generic_url and not validators.url(generic_url):
#         st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
#     else:
#         try:
#             with st.spinner("Processing..."):
#                 docs = []

#                 # Process YouTube or Website URL
#                 if generic_url:
#                     if "youtube.com" in generic_url or "youtu.be" in generic_url:
#                         try:
#                             video_id = (
#                                 generic_url.split("v=")[-1].split("&")[0]
#                                 if "youtube.com" in generic_url
#                                 else generic_url.split("/")[-1]
#                             )
#                             transcript = YouTubeTranscriptApi.get_transcript(video_id)
#                             content = " ".join([t['text'] for t in transcript])
#                             docs.append(Document(page_content=content, metadata={"source": generic_url}))
#                         except TranscriptsDisabled:
#                             st.error("Transcripts are disabled for this video. Please try another video.")
#                         except Exception as e:
#                             st.error(f"Failed to fetch YouTube content: {e}")
#                     else:
#                         loader = UnstructuredURLLoader(
#                             urls=[generic_url],
#                             ssl_verify=False,
#                             headers={
#                                 "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
#                             },
#                         )
#                         unstructured_docs = loader.load()
#                         for doc in unstructured_docs:
#                             content, token_count = truncate_content(doc.page_content, MAX_TOKENS - TOKEN_RESERVE)
#                             docs.append(Document(page_content=content, metadata={"source": generic_url}))
#                             if token_count > MAX_TOKENS:
#                                 st.warning(f"Content was truncated to {MAX_TOKENS - TOKEN_RESERVE} tokens.")

#                 # Process Audio File
#                 if audio_file:
#                     # Save the uploaded file temporarily
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as temp_audio_file:
#                         temp_audio_file.write(audio_file.read())
#                         temp_audio_path = temp_audio_file.name

#                     # Load and transcribe the saved audio file
#                     model = whisper.load_model("base")
#                     try:
#                         transcription = model.transcribe(temp_audio_path)
#                         audio_content = transcription['text']
#                         docs.append(Document(page_content=audio_content, metadata={"source": "Uploaded Audio"}))
#                     except Exception as e:
#                         st.error(f"Failed to transcribe audio: {e}")
#                         raise

#                 # Handle Empty or Invalid Data
#                 if not docs:
#                     st.error("No content found. Please check the input and try again.")
#                 else:
#                     # Summarization Chain
#                     chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
#                     output_summary = chain.run(docs)

#                     # Display the Summary
#                     st.success(output_summary)
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#             st.exception(e)

# Working version of youtube and website 
# # app.py

# import validators
# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.chains.summarize import load_summarize_chain
# from langchain.schema import Document
# from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api._errors import TranscriptsDisabled


# # Streamlit App Configuration
# st.set_page_config(
#     page_title="LangChain: Summarize Text From YT or Website",
#     page_icon="🦜",
# )
# st.title("🦜 LangChain: Summarize Text From YT or Website")
# st.subheader('Summarize URL')

# # Sidebar for Groq API Key Input
# with st.sidebar:
#     groq_api_key = st.text_input("Groq API Key", value="", type="password")

# # URL Input Field
# generic_url = st.text_input("URL", label_visibility="collapsed")

# # LangChain LLM Setup (Gemma Model)
# llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# # Prompt Template
# prompt_template = """
# Provide a summary of the following content in 300 words:
# Content:{text}
# """
# prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# # Token limit
# MAX_TOKENS = 15000  # The maximum tokens allowed by the model
# TOKEN_RESERVE = 1000  # Reserve tokens for system prompts and metadata

# def truncate_content(content, max_tokens):
#     """Truncate content to fit within the token limit."""
#     tokens = content.split()
#     if len(tokens) > max_tokens:
#         truncated = " ".join(tokens[:max_tokens])
#         return truncated, len(tokens)
#     return content, len(tokens)

# # Button for Summarization
# if st.button("Summarize the Content from YT or Website"):
#     # Validate Inputs
#     if not groq_api_key.strip() or not generic_url.strip():
#         st.error("Please provide the information to get started.")
#     elif not validators.url(generic_url):
#         st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
#     else:
#         try:
#             with st.spinner("Waiting..."):
#                 # Load content from the URL
#                 docs = []
#                 if "youtube.com" in generic_url or "youtu.be" in generic_url:
#                     try:
#                         video_id = (
#                             generic_url.split("v=")[-1].split("&")[0]
#                             if "youtube.com" in generic_url
#                             else generic_url.split("/")[-1]
#                         )
#                         transcript = YouTubeTranscriptApi.get_transcript(video_id)
#                         content = " ".join([t['text'] for t in transcript])
#                         docs.append(Document(page_content=content, metadata={"source": generic_url}))
#                     except TranscriptsDisabled:
#                         st.error("Transcripts are disabled for this video. Please try another video.")
#                     except Exception as e:
#                         st.error(f"Failed to fetch YouTube content: {e}")
#                 else:
#                     loader = UnstructuredURLLoader(
#                         urls=[generic_url],
#                         ssl_verify=False,
#                         headers={
#                             "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
#                         },
#                     )
#                     unstructured_docs = loader.load()
#                     for doc in unstructured_docs:
#                         content, token_count = truncate_content(doc.page_content, MAX_TOKENS - TOKEN_RESERVE)
#                         docs.append(Document(page_content=content, metadata={"source": generic_url}))
#                         if token_count > MAX_TOKENS:
#                             st.warning(f"Content was truncated to {MAX_TOKENS - TOKEN_RESERVE} tokens.")

#                 # Handle Empty or Invalid Data
#                 if not docs:
#                     st.error("No content found. Please check the URL and try again.")
#                 else:
#                     # Summarization Chain
#                     chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
#                     output_summary = chain.run(docs)

#                     # Display the Summary
#                     st.success(output_summary)
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#             st.exception(e)



