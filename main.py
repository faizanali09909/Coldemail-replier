import os
# Disable CrewAI telemetry BEFORE any crewai imports to avoid signal handler errors
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import streamlit as st
from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
from crewai_tools import ScrapeWebsiteTool
import time

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Cold Email Generator Pro",
    page_icon="✨",
    layout="wide"
)

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
st.sidebar.info("🚀 Powered by Groq Llama-3.3-70b")
user_model = "groq/llama-3.3-70b-versatile"
st.sidebar.markdown("---")

# User Input Section
st.markdown("### 🎯 Target Profile & Details")
col_url, col_name, col_company = st.columns(3)
with col_url:
    target_url = st.text_input(
        "🏢 Website to analyze:",
        placeholder="e.g., https://openai.com/",
    )
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

# Agency Services Knowledge Base (Read-only)
AGENCY_SERVICES = """1. Web Scraping: Extracting data from websites for various purposes such as market research, competitive analysis, or content aggregation.
2. Data Analysis: Analyzing data to uncover patterns, trends, and insights that can inform business decisions.
3. Content Generation: Creating written content such as articles, blog posts, or social media updates"""

with st.expander("🛠️ Agency Services Knowledge Base"):
    st.markdown("### 📋 What services we offer")
    st.info(AGENCY_SERVICES)
    # Using a hidden or internal variable for the crew to use
    agency_services = AGENCY_SERVICES

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
    elif not user_name:
         st.warning("⚠️ Please enter your Name!")
    elif not user_company:
         st.warning("⚠️ Please enter your Company!")
    else:
        with st.spinner("🤖 Our AI Agents are currently scraping the site, strategizing, and writing the perfect email..."):
            try:
                # Initialize Tools and LLM
                scrape_tool = ScrapeWebsiteTool()
                
                llm = LLM(
                    model=user_model,
                    api_key=os.getenv("GROQ_API_KEY")
                )

                # Initialize Agents
                researcher = Agent(
                    role="Research on the Targetted Website",
                    goal="Scrape the website and extract the information needed",
                    backstory="You have researched many of the Websites",
                    llm=llm,
                    tools=[scrape_tool],
                    verbose=False,
                    allow_delegation=True,
                    memory=True,
                )

                analyst = Agent(
                    role="Analyze the data get from the targetted URL",
                    goal="Analyze the data in website and write in short form",
                    backstory="You are a great analyst that can analyze the data and give me the insights",
                    llm=llm,
                    verbose=False,
                    memory=True,
                )

                writer = Agent(
                    role="Write a personalized cold email",
                    goal="Create a compelling cold email based on the analyzed data",
                    backstory="You are a skilled copywriter with experience in creating effective sales emails",
                    llm=llm,
                    verbose=True,
                    memory=True,
                )

                # Define Tasks
                task_analyze = Task(
                    description=f"Scrape the website {target_url}. Summarize what the company does and identify 1 key area where they could improve (e.g., design, traffic, automation).",
                    expected_output="A brief summary of the company and their potential pain points.",
                    agent=researcher
                )

                task_strategize = Task(
                    description=f"Based on the analysis, pick ONE service from our Agency Knowledge Base that solves their problem. Explain the match.\nAgency Knowledge Base:\n{agency_services}",
                    expected_output="The selected service and the reasoning for the match.",
                    agent=analyst
                )

                task_write = Task(
                    description=f"Draft a cold email to the CEO of the target company. Pitch the selected service. Keep it under 150 words. Ensure the email is signed off by '{user_name}' from '{user_company}'.",
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
    <p style='color: #94a3b8; font-size: 0.8rem; margin-top: 5px;'>Powered by CrewAI & Groq (Llama 3.3 70B Versatile) 🚀</p>
</div>
""", unsafe_allow_html=True)