import streamlit as st
import os, requests, pytz
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

# 1. LIGHTWEIGHT INITIALIZATION
load_dotenv()
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
except:
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Reliable AI Agent", page_icon="🎯", layout="wide")

# 2. THE TRUTH ENGINE (Instant & Local)
def get_verified_time():
    # Forced Malaysia Time to stop the '05:33 AM' hallucination
    myt = pytz.timezone("Asia/Kuala_Lumpur")
    return datetime.now(myt).strftime('%I:%M %p, %A, %B %d, %Y')

def get_live_weather(city):
    try:
        # Fast weather check without heavy libraries
        return requests.get(f"https://wttr.in/{city}?format=3", timeout=3).text
    except:
        return "Weather service temporarily unavailable."

# 3. CHAT STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Using 8B-Instant for the fastest response time
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, streaming=True)

# 4. SIDEBAR CONFIG
with st.sidebar:
    st.title("⚙️ Reliability Settings")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    user_city = st.text_input("Current City:", "Kuala Lumpur")
    st.markdown("---")
    if st.button("🗑️ Reset All Memory"):
        st.session_state.chat_history = []
        st.rerun()

    persona_prompts = {
        "Professional": "You are a factual, elite assistant. Accuracy is your #1 priority.",
        "Sassy": "You are witty and sarcastic, but you NEVER lie about facts. 💅",
        "Emo": "You are moody and deep. You see the cold truth in everything. 🖤"
    }

st.title(f"🤖 {persona} Assistant")

# Display previous messages
for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 5. THE GROUNDED RESPONSE LOOP
if user_input := st.chat_input("Ask a fact-based question..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        # A. Fetch Truths
        curr_time = get_verified_time()
        
        # B. Smart Tool Triggering
        # Only search if the user asks for news, people, businesses, or weather
        weather_data = ""
        search_results = ""
        
        if any(k in user_input.lower() for k in ["weather", "temp", "rain", "hot"]):
            with st.status("Checking local weather...", expanded=False):
                weather_data = get_live_weather(user_city)
        
        # Fact-check keywords trigger a deeper web search
        fact_keywords = ["who", "what is", "news", "latest", "today", "restaurant", "brand", "price"]
        if any(k in user_input.lower() for k in fact_keywords) and tavily_api_key:
            with st.status("Verifying facts on the web...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                # Advanced search helps avoid the 'Naknak is Malay food' error
                search_results = str(tavily.search(query=user_input, search_depth="advanced", max_results=3))

        # C. THE FORCEFUL SYSTEM PROMPT
        # This prevents the 'I don't have access' lie and the '05:33 AM' hallucination.
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            f"MANDATORY CONTEXT:\n"
            f"- Current Location: {user_city}\n"
            f"- Verified Time: {curr_time}\n"
            f"- Local Weather: {weather_data}\n"
            f"- Web Search Data: {search_results}\n\n"
            "RULES FOR RELIABILITY:\n"
            "1. Use the 'Verified Time' above for all time questions. Do not argue with it.\n"
            "2. If 'Web Search Data' says something (e.g., 'Naknak is Korean'), you MUST use that info. "
            "Internal training is often outdated; Web Search Data is the absolute truth.\n"
            "3. NEVER say 'I don't have real-time access'. You DO have it via the context provided.\n"
            "4. Only mention the weather/time if the user specifically asks for it.\n"
            "5. If you cannot find info in the Search Data, say 'I couldn't verify that' instead of guessing."
        )

        # D. High-Speed Streaming
        full_response = st.write_stream(llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(content=full_response))
