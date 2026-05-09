import os
import sys
import re
# Disable CrewAI telemetry BEFORE any crewai imports to avoid signal handler errors
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Fix for Streamlit Cloud SQLite version requirement by ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Cold Email Generator Pro",
    page_icon="✨",
    layout="wide"
)

# Diagnostic block to find the hidden ImportError
try:
    from crewai import Agent, Task, Crew, LLM, Process
    from crewai.tools import tool
    from crewai_tools import ScrapeWebsiteTool
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import chromadb
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.info("💡 Please ensure all requirements are installed in your environment.")
    st.code("pip install crewai crewai-tools langchain-community langchain-google-genai chromadb pypdf pysqlite3-binary")
    st.stop()

from dotenv import load_dotenv
import time
from typing import Any
import tempfile
import re

# Load environment variables
load_dotenv()

# Streamlit Cloud uses st.secrets, let's sync it to os.environ if missing
try:
    if "GEMINI_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# Ensure GOOGLE_API_KEY is set for native provider support
api_key = os.getenv("GEMINI_API_KEY", "")
if not os.getenv("GOOGLE_API_KEY") and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Global Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header Styling */
    .main-header {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        margin-bottom: 0px;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4), 0 2px 4px -1px rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5), 0 4px 6px -2px rgba(59, 130, 246, 0.3);
    }
    
    /* Result Box / Transparent Styling */
    .result-box {
        padding: 2rem;
        border-radius: 16px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        white-space: pre-wrap;
        color: #1e293b;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Data Input styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    
    /* Custom Footer Box */
    .footer-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .footer-text {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.2rem;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main Container
col_padding1, main_content, col_padding2 = st.columns([1, 6, 1])

with main_content:
    # Header
    st.markdown("<h1 class='main-header'>✨ Cold Email Generator Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Generate highly-personalized cold emails for your prospects using AI Agents in seconds</p>", unsafe_allow_html=True)

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'result' not in st.session_state:
    st.session_state.result = None

# Sidebar Configuration
st.sidebar.markdown("<h2 style='text-align: center; color: #1e3a8a;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
st.sidebar.info("🚀 Powered by Gemini 2.0 Flash Lite")
st.sidebar.markdown(f"<p style='font-size: 0.8rem; color: #64748b;'>Python: {os.path.basename(sys.executable)}</p>", unsafe_allow_html=True)
try:
    import google.genai
    st.sidebar.success("✅ Provider Found")
except ImportError:
    st.sidebar.error("❌ Provider Missing")
user_model = "gemini/gemini-flash-latest"
st.sidebar.markdown("---")

# User Input Section
st.markdown("### 🎯 Target Profile & Details")
col_url, col_owner = st.columns(2)
with col_url:
    target_url = st.text_input(
        "🏢 Website to analyze:",
        placeholder="e.g., https://openai.com/",
    )
with col_owner:
    target_owner = st.text_input(
        "👑 Target Owner Name:",
        placeholder="e.g., Sam Altman"
    )

col_name, col_company = st.columns(2)
with col_name:
    user_name = st.text_input(
        "👤 Your Name:",
        placeholder="e.g., John Doe"
    )
with col_company:
    user_company = st.text_input(
        "💼 Your Company:",
        placeholder="e.g., Acme Corp"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Knowledge Base Tool
# Knowledge Base Logic
knowledge_file = st.file_uploader("Upload Agency Profile (PDF or TXT) to customize knowledge:", type=["pdf", "txt"])

AGENCY_SERVICES_DEFAULT = """1. Web Scraping: Extracting data from websites for various purposes.
2. Data Analysis: Analyzing data to uncover patterns and insights.
3. Content Generation: Creating high-quality written content using AI."""

vectorstore = None

if knowledge_file:
    with st.spinner("📚 Indexing your knowledge base..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{knowledge_file.name.split('.')[-1]}") as tmp:
            tmp.write(knowledge_file.getvalue())
            tmp_path = tmp.name
        
        try:
            if knowledge_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
            
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
            chunks = splitter.split_documents(docs)
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
            vectorstore = Chroma.from_documents(chunks, embeddings)
            st.success("✅ Knowledge base indexed successfully!")
        except Exception as e:
            st.error(f"Failed to index file: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

with st.expander("📋 Current Knowledge Base Preview"):
    if vectorstore:
        st.info("Using uploaded documents for intelligence.")
    else:
        st.info(AGENCY_SERVICES_DEFAULT)
        agency_services = AGENCY_SERVICES_DEFAULT

st.markdown("<br>", unsafe_allow_html=True)

# Process Button
colact1, colact2, colact3 = st.columns([1, 2, 1])
with colact2:
    process_clicked = st.button("🚀 Generate Winning Email", use_container_width=True)

def clean_output(text):
    # Remove phrases like "Thought: I now can give a great answer."
    patterns = [
        r"^Thought:\s*I now can give a great answer\.\s*",
        r"^Thought:\s*I've got the final answer\s*",
        r"^Final Answer:\s*"
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def run_crew_with_retry(crew, inputs=None, max_retries=5):
    """Run crew with retry logic for handling rate limit and 503 errors"""
    for attempt in range(max_retries):
        try:
            return crew.kickoff(inputs=inputs) if inputs else crew.kickoff()
        except Exception as e:
            error_str = str(e)
            # Handle 503 - Service Unavailable
            if "503" in error_str or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    st.warning(f"🤖 Model busy, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e
            # Handle 429 - Rate Limit Exceeded
            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    # Extract retry delay from error if available, otherwise use default
                    wait_time = 20  # Default 20 seconds for rate limits
                    if "retryDelay" in error_str:
                        try:
                            # Try to parse the retry delay from error
                            import re
                            match = re.search(r'(\d+)s', error_str)
                            if match:
                                wait_time = int(match.group(1)) + 2  # Add buffer
                        except:
                            pass
                    st.warning(f"⏱️ Rate limit hit! Waiting {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e
            else:
                raise e

if process_clicked:
    if not target_url:
         st.warning("⚠️ Please enter a Target Company URL first!")
    elif not target_url.startswith("http"):
         st.warning("⚠️ Please provide a valid URL starting with http:// or https://")
    elif not target_owner:
         st.warning("⚠️ Please enter the Target Owner Name!")
    elif not user_name:
         st.warning("⚠️ Please enter your Name!")
    elif not user_company:
         st.warning("⚠️ Please enter your Company!")
    elif not os.getenv("GEMINI_API_KEY"):
         st.error("🔑 API Key Missing! Please add your GEMINI_API_KEY to Streamlit Cloud secrets (Settings -> Secrets) or in your local .env file.")
    else:
        with st.spinner("🤖 Our AI Agents are currently scraping the site, strategizing, and writing the perfect email..."):
            try:
                # Initialize Tools and LLM
                scrape_tool = ScrapeWebsiteTool(config=dict(cert_verify=False))
                
                if vectorstore:
                    @tool("Search Agency Knowledge")
                    def knowledge_tool(query: str):
                        """Search the agency's internal documents for services, case studies, and expertise."""
                        results = vectorstore.similarity_search(query, k=2)
                        return "\n".join([r.page_content for r in results])
                else:
                    knowledge_tool = None
                
                llm = LLM(
                    model=user_model,
                    api_key=os.getenv("GEMINI_API_KEY")
                )

                # Initialize Agents
                researcher = Agent(
                    role="Website Researcher",
                    goal=f"Scrape {target_url} and identify core business and pain points.",
                    backstory="Expert at analyzing company websites to find opportunities for improvement.",
                    llm=llm,
                    tools=[scrape_tool],
                    verbose=False
                )

                analyst = Agent(
                    role="Service Strategist",
                    goal="Find the best matching service from our knowledge base for the prospect's needs.",
                    backstory="Strategic consultant who excels at matching company problems with agency solutions.",
                    llm=llm,
                    tools=[knowledge_tool] if knowledge_tool else [],
                    verbose=False
                )

                writer = Agent(
                    role="Email Copywriter",
                    goal="Write a persuasive, short, and personalized cold email.",
                    backstory="World-class cold email expert who writes high-conversion pitches.",
                    llm=llm,
                    verbose=True
                )

                # Define Tasks
                task_analyze = Task(
                    description=f"Scrape the website {target_url}. Summarize what the company does and identify 1 key area where they could improve (e.g., design, traffic, automation).",
                    expected_output="A brief summary of the company and their potential pain points.",
                    agent=researcher
                )

                knowledge_source = "Search our knowledge base using your tool." if knowledge_tool else f"Use the following static services list:\n{agency_services}"
                task_strategize = Task(
                    description=f"Based on the analysis, pick ONE service that solves their problem. {knowledge_source}",
                    expected_output="The selected service and the reasoning for the match.",
                    agent=analyst
                )

                task_write = Task(
                    description=f"Draft a cold email to {target_owner}. Pitch the selected service based on their website '{target_url}'. Keep it under 150 words. Address the email directly to {target_owner}. Ensure the email is signed off by '{user_name}' from '{user_company}'.",
                    expected_output="A professional cold email ready to send.",
                    agent=writer
                )

                # Create Crew
                sales_crew = Crew(
                    agents=[researcher, analyst, writer],
                    tasks=[task_analyze, task_strategize, task_write],
                    process=Process.sequential,
                    verbose=False
                )

                # Execute with retry
                raw_result = run_crew_with_retry(sales_crew)
                result = clean_output(str(raw_result))
                st.session_state.result = result
                
                # Save to history
                # Extract a clean name from the URL
                display_name = target_url.replace("https://", "").replace("http://", "").replace("www.", "").split('/')[0].split('.')[0].capitalize()
                
                st.session_state.history.append({
                    "name": display_name,
                    "url": target_url,
                    "email": result
                })
                
                st.success("✨ Email generated successfully!")
                
                # Display Result Immediately Let's put it right here below success.
                st.markdown("---")
                st.markdown("<h3 style='color: #1e3a8a; text-align: center;'>💌 Your Highly-Personalized Cold Email</h3>", unsafe_allow_html=True)
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(result)
                st.markdown("</div>", unsafe_allow_html=True)
            
                # Download button for latest
                result_text = str(result)
                
                dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
                with dl_col2:
                    st.download_button(
                        label="📥 Download This Email",
                        data=result_text,
                        file_name="cold_email.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    st.error("🤖 **Model Currently Busy**. Please try again shortly.")
                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    st.error("⏱️ **Rate Limit Exceeded**. Please wait a minute before trying again.")
                else:
                    st.error(f"An error occurred: {str(e)}")


# Display History in Sidebar
if st.session_state.history:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕰️ Your History")
    
    # Add Clear History Button
    if st.sidebar.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.result = None
        st.rerun()

    for i, item in enumerate(reversed(st.session_state.history)):
        # Display as name instead of URL link
        with st.sidebar.expander(f"Email for {item.get('name', 'Company')} ({len(st.session_state.history) - i})"):
            st.markdown(item["email"])

# Footer
st.markdown("""
<div class='footer-box'>
    <p class='footer-text'>✨ This website is made by Faizan Ali ✨</p>
    <p style='color: #94a3b8; font-size: 0.8rem; margin-top: 5px;'>Powered by CrewAI & Google (Gemini 2.0 Flash Lite) 🚀</p>
</div>
""", unsafe_allow_html=True)