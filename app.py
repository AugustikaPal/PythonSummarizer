# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.chains.summarize import load_summarize_chain
# from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
# from bs4 import BeautifulSoup
# from langchain.schema import Document
# import validators
# import requests
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import re
# import time

# # Set up the Streamlit app configuration
# st.set_page_config(page_title="Reference Content Summarizer", page_icon="📚", layout="wide")

# # Page title with enhanced formatting
# st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📚 Reference Content Summarizer</h1>", unsafe_allow_html=True)

# # Sidebar for Groq API Key
# with st.sidebar:
#     st.header("🔧 Settings")
#     groq_api_key = st.text_input("Groq API Key", value="", type="password", help="Enter your Groq API key to access the summarization service.")

# # Input for topic or title with instructions
# st.subheader("📌 Enter the Topic or Title")
# topic = st.text_input("Topic/Title", placeholder="e.g., Importance of Discipline", help="Provide a clear and specific topic for better summaries.")

# # Section for YouTube video URLs
# st.subheader("🎥 Enter YouTube Video URLs")
# video_urls = st.text_area("YouTube Video URLs (one per line)", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ", help="Enter one YouTube URL per line.")

# # Section for website URLs
# st.subheader("🌐 Enter Website URLs")
# website_urls = st.text_area("Website URLs (one per line)", placeholder="e.g., https://www.example.com", help="Enter one website URL per line.")

# # Section for PDF file uploads
# st.subheader("📄 Upload PDFs")
# uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, help="Upload PDF documents relevant to your topic.")

# # Summary length customization
# st.sidebar.subheader("📏 Summary Length")
# summary_length = st.sidebar.slider("Choose summary length (words)", min_value=100, max_value=3000, value=1000, format="%d", step=100, help="Adjust the length of the summary in words.")

# # Initialize session state history if not already done
# if "history" not in st.session_state:
#     st.session_state.history = {}

# # Helper function to fetch and parse website content
# def fetch_website_content(url):
#     try:
#         response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
#         response.raise_for_status()
#         response.encoding = 'utf-8'
#         soup = BeautifulSoup(response.text, 'html.parser')
#         paragraphs = soup.find_all('p')
#         content = "\n".join([para.get_text() for para in paragraphs])
#         return content
#     except Exception as e:
#         st.error(f"⚠️ Error fetching content from {url}: {e}")
#         return None

# # Helper function to clean text
# def clean_text(text):
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#     return text

# # Helper function to determine relevance of content
# def is_relevant(text, topic):
#     topic_keywords = topic.lower().split()
#     text = text.lower()
#     return any(keyword in text for keyword in topic_keywords)

# # Improved button with primary color
# if st.button("✨ Generate Summary"):
#     if not groq_api_key.strip():
#         st.error("❗ Please provide the API key.")
#         st.stop()

#     llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

#     if not topic.strip():
#         st.error("❗ Please provide a topic or title.")
#         st.stop()

#     source_summaries = {}
#     # Process YouTube Videos
#     if video_urls.strip():
#         video_urls_list = [url.strip() for url in video_urls.split("\n") if validators.url(url) and "youtube.com" in url]
#         for url in video_urls_list:
#             try:
#                 loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
#                 docs = loader.load()
#                 cleaned_docs = [Document(page_content=clean_text(doc.page_content)) for doc in docs]
#                 if cleaned_docs:
#                     source_summaries[url] = cleaned_docs
#             except Exception as e:
#                 st.error(f"⚠️ Error processing {url}: {str(e)}")

#     # Process Website URLs
#     if website_urls.strip():
#         website_urls_list = [url.strip() for url in website_urls.split("\n") if validators.url(url)]
#         for url in website_urls_list:
#             content = fetch_website_content(url)
#             if content:
#                 cleaned_content = clean_text(content)
#                 source_summaries[url] = [Document(page_content=cleaned_content)]

#     # Process PDFs
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             temp_pdf = f"./temp.pdf"
#             with open(temp_pdf, "wb") as file:
#                 file.write(uploaded_file.getvalue())

#             loader = PyPDFLoader(temp_pdf)
#             docs = loader.load()
#             source_summaries[uploaded_file.name] = docs

#     if source_summaries:
#         st.subheader("📝 Summarized Content")

#         # Set up tabs for each source
#         tabs = st.tabs(list(source_summaries.keys()))
#         for i, (source, docs) in enumerate(source_summaries.items()):
#             with tabs[i]:
#                 with st.spinner("Processing content..."):
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#                     splits = text_splitter.split_documents(docs)

#                     prompt_template = """
#                     Given the topic "{topic}", provide a summarized overview of the key information relevant to the topic from the following content:
#                     Content: {text}
#                     """
#                     prompt = PromptTemplate(template=prompt_template, input_variables=["text", "topic"])

#                     summaries = []
#                     chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

#                     for split in splits:
#                         doc_text = clean_text(split.page_content)
#                         summary = chain.run(input_documents=[Document(page_content=doc_text)], topic=topic)

#                         # Only include summaries if they are relevant
#                         if is_relevant(summary, topic):
#                             summaries.append(summary)

#                     # Check summary length and handle shortfall
#                     full_summary = "\n\n".join(summaries)
#                     summary_word_count = len(full_summary.split())

#                     if summary_word_count < summary_length:
#                         st.warning(f"⚠️ Content from {source} is shorter than the selected summary length ({summary_length} words). Providing available summary.")

#                     if full_summary:
#                         st.text_area(f"Summary for {source}", value=full_summary, height=300, disabled=True)
#                         # Store summary under topic/title in history
#                         st.session_state.history[topic] = full_summary
#                     else:
#                         st.warning(f"No relevant content found from {source}.")
#     else:
#         st.warning("⚠️ No valid content was found to summarize.")

# # Display history of summaries by topic/title
# if st.session_state.history:
#     st.sidebar.subheader("🗂️ Summary History")
#     selected_topic = st.sidebar.selectbox("Select a topic to view summary", options=list(st.session_state.history.keys()))

#     if selected_topic:
#         st.sidebar.markdown(f"**{selected_topic}**")
#         st.sidebar.text_area("Summary", value=st.session_state.history[selected_topic], height=150, disabled=True)
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from bs4 import BeautifulSoup
from langchain.schema import Document
import validators
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import time

# Set up the Streamlit app configuration
st.set_page_config(page_title="Reference Content Summarizer", page_icon="📚", layout="wide")

# Page title with enhanced formatting
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📚 Reference Content Summarizer</h1>", unsafe_allow_html=True)

# Sidebar for Groq API Key
with st.sidebar:
    st.header("🔧 Settings")
    groq_api_key = st.text_input("Groq API Key", value="", type="password", help="Enter your Groq API key to access the summarization service.")

# Input for topic or title with instructions
st.subheader("📌 Enter the Topic or Title")
topic = st.text_input("Topic/Title", placeholder="e.g., Importance of Discipline", help="Provide a clear and specific topic for better summaries.")

# Section for YouTube video URLs
st.subheader("🎥 Enter YouTube Video URLs")
video_urls = st.text_area("YouTube Video URLs (one per line)", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ", help="Enter one YouTube URL per line.")

# Section for website URLs
st.subheader("🌐 Enter Website URLs")
website_urls = st.text_area("Website URLs (one per line)", placeholder="e.g., https://www.example.com", help="Enter one website URL per line.")

# Section for PDF file uploads
st.subheader("📄 Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, help="Upload PDF documents relevant to your topic.")

# Summary length customization
st.sidebar.subheader("📏 Summary Length")
summary_length = st.sidebar.slider("Choose summary length (words)", min_value=100, max_value=3000, value=1000, format="%d", step=100, help="Adjust the length of the summary in words.")

# Initialize session state history if not already done
if "history" not in st.session_state:
    st.session_state.history = {}

# Helper function to fetch and parse website content
def fetch_website_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        st.error(f"⚠️ Error fetching content from {url}: {e}")
        return None

# Helper function to clean text
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

# Helper function to determine relevance of content
def is_relevant(text, topic):
    topic_keywords = topic.lower().split()
    text = text.lower()
    return any(keyword in text for keyword in topic_keywords)

# Improved button with primary color
if st.button("✨ Generate Summary"):
    if not groq_api_key.strip():
        st.error("❗ Please provide the API key.")
        st.stop()

    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

    if not topic.strip():
        st.error("❗ Please provide a topic or title.")
        st.stop()

    source_summaries = {}
    # Process YouTube Videos
    if video_urls.strip():
        video_urls_list = [url.strip() for url in video_urls.split("\n") if validators.url(url) and "youtube.com" in url]
        for url in video_urls_list:
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                docs = loader.load()
                cleaned_docs = [Document(page_content=clean_text(doc.page_content)) for doc in docs]
                if cleaned_docs:
                    source_summaries[url] = cleaned_docs
            except Exception as e:
                st.error(f"⚠️ Error processing {url}: {str(e)}")

    # Process Website URLs
    if website_urls.strip():
        website_urls_list = [url.strip() for url in website_urls.split("\n") if validators.url(url)]
        for url in website_urls_list:
            content = fetch_website_content(url)
            if content:
                cleaned_content = clean_text(content)
                source_summaries[url] = [Document(page_content=cleaned_content)]

    # Process PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            source_summaries[uploaded_file.name] = docs

    if source_summaries:
        st.subheader("📝 Summarized Content")

        # Set up tabs for each source
        tabs = st.tabs(list(source_summaries.keys()))
        for i, (source, docs) in enumerate(source_summaries.items()):
            with tabs[i]:
                with st.spinner("Processing content..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                    splits = text_splitter.split_documents(docs)

                    prompt_template = """
                    Given the topic "{topic}", provide a summarized overview of the key information relevant to the topic from the following content:
                    Content: {text}
                    """
                    prompt = PromptTemplate(template=prompt_template, input_variables=["text", "topic"])

                    summaries = []
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

                    for split in splits:
                        doc_text = clean_text(split.page_content)
                        summary = chain.run(input_documents=[Document(page_content=doc_text)], topic=topic)

                        # Only include summaries if they are relevant
                        if is_relevant(summary, topic):
                            summaries.append(summary)

                    # Check summary length and handle shortfall
                    full_summary = "\n\n".join(summaries)
                    summary_word_count = len(full_summary.split())

                    if summary_word_count < summary_length:
                        st.warning(f"⚠️ Content from {source} is shorter than the selected summary length ({summary_length} words). Providing available summary.")

                    if full_summary:
                        st.text_area(f"Summary for {source}", value=full_summary, height=300, disabled=True)
                        # Store summary under topic/title in history
                        st.session_state.history[topic] = full_summary
                    else:
                        st.warning(f"No relevant content found from {source}.")
    else:
        st.warning("⚠️ No valid content was found to summarize.")

# Display history of summaries by topic/title
if st.session_state.history:
    st.sidebar.subheader("🗂️ Summary History")
    
    # Display each topic as a clickable button
    for topic in st.session_state.history.keys():
        if st.sidebar.button(topic):
            st.markdown(f"### {topic}")
            st.text_area("Summary", value=st.session_state.history[topic], height=150, disabled=True)
