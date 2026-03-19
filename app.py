import streamlit as st
import os, pytz
import torch
import time
from transformers import pipeline
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

# Load local environment variables if they exist
load_dotenv()

# --- 1. CONFIG & API SETUP ---
st.set_page_config(page_title="Global Hybrid AI", page_icon="🌍", layout="wide")

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")

# --- 2. LOCAL PYTORCH OPTIMIZATION ---
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

analyzer = load_sentiment_model()

# --- 3. UTILITIES: TIME & HISTORY ---
def get_device_time():
    try:
        user_tz_name = st.context.timezone or "UTC"
        user_tz = pytz.timezone(user_tz_name)
    except:
        user_tz = pytz.timezone("Asia/Kuala_Lumpur") 
        user_tz_name = "Asia/Kuala_Lumpur"
    
    now = datetime.now(user_tz)
    return now.strftime('%I:%M %p, %A, %B %d, %Y'), user_tz_name

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. SIDEBAR (REFINED) ---
with st.sidebar:
    st.title("⚙️ AI Logic Center")
    
    # NEW: Model Selection for 2026 Models
    st.subheader("🤖 Brain Selection")
    selected_model_id = st.selectbox(
        "Choose Groq Engine:",
        [
            "openai/gpt-oss-120b",           # High Reasoning
            "meta-llama/llama-4-scout-17b-16e-instruct", # Ultra-Fast/Multimodal
            "meta-llama/llama-3.3-70b-versatile", # Reliable Baseline
            "qwen/qwen3-32b"                 # Best for Multilingual
        ],
        index=1,
        help="GPT-OSS is best for logic; Llama 4 Scout is fastest."
    )
    
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    
    st.divider()
    st.subheader("🧠 Local PyTorch Engine")
    device_info = "GPU (Accelerated)" if torch.cuda.is_available() else "CPU (Standard)"
    st.info(f"Inference Mode: **{device_info}**")
    st.caption("DistilBERT analyzing user intent locally.")

    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

    persona_prompts = {
        "Professional": "You are a factual, elite assistant. Be polite and precise.",
        "Sassy": "You are witty and sarcastic. Be funny but helpful.",
        "Emo": "You are moody and deep. Everything is gray and meaningless."
    }

# Initialize LLM with the selected model
llm = ChatGroq(
    model=selected_model_id, 
    api_key=groq_api_key, 
    streaming=True, 
    temperature=0
)

# --- 5. CHAT INTERFACE ---
st.title(f"🤖 {persona} Grounded Assistant")
st.caption(f"Currently powered by: **{selected_model_id}**")

# Display History
for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# Input Logic
if user_input := st.chat_input("Ask me anything..."):
    # Step A: Pre-processing with Local PyTorch
    with st.spinner("Analyzing sentiment..."):
        analysis = analyzer(user_input)[0]
        user_mood = analysis['label']
        mood_score = round(analysis['score'], 2)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Step B: Response Generation
    with st.chat_message("assistant"):
        curr_time, tz_name = get_device_time()
        
        # Determine if we need to RAG (Search)
        needs_fact_check = any(k in user_input.lower() for k in [
            "who", "what", "where", "news", "price", "is it", "weather", "time in", "studios", "policy", "location"
        ])
        
        search_data = "NO_EXTERNAL_SEARCH_RESULTS_FOUND"
        sources_text = ""
        
        if needs_fact_check and tavily_api_key:
            with st.status("🔍 Searching live database...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                query = f"{user_input} latest info 2026"
                response = tavily.search(query=query, search_depth="advanced", max_results=4)
                
                search_data = "\n\n".join([
                    f"--- DOCUMENT {i+1} ---\nSource: {res['title']}\nSnippet: {res['content']}" 
                    for i, res in enumerate(response['results'])
                ])
                sources_text = "\n".join([f"- [{i+1}] {res['title']}: {res['url']}" for i, res in enumerate(response['results'])])

        # --- THE GROUNDING PROMPT ---
        sys_msg = (
            f"SYSTEM ROLE: {persona_prompts[persona]}\n"
            f"USER_METADATA: Sentiment={user_mood}, Timezone={tz_name}, LocalTime={curr_time}\n\n"
            f"PROVIDED_CONTEXT:\n{search_data}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer ONLY using the PROVIDED_CONTEXT. Do not use outside knowledge.\n"
            f"2. If the context does not contain the answer, say: 'I don't have enough specific data to answer that accurately.'\n"
            f"3. Use inline citations like [1] or [2] next to facts extracted from the context.\n"
            f"4. Maintain your persona but prioritize FACTUAL ACCURACY over creativity.\n"
            f"5. If asked about time, refer to USER_METADATA LocalTime."
        )

        # Start timer for performance tracking
        start_time = time.time()

        # Generate and Stream
        full_response = st.write_stream(
            llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history)
        )
        
        duration = round(time.time() - start_time, 2)
        
        # Display Performance & Sources
        st.caption(f"⏱️ Inference: {duration}s | 🧠 Sentiment: {user_mood} ({mood_score})")
        
        if sources_text:
            with st.expander("📚 View Sources"):
                st.markdown(sources_text)
            full_response += f"\n\nSources Found:\n{sources_text}"

        st.session_state.chat_history.append(AIMessage(content=full_response))
