import os
import base64
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama # For local PC mode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from tools import all_tools
from database import index_any_csv, search_data_vault, index_text_snippet

app = FastAPI(title="Global Vision AI: Hybrid Edition")

# --- HYBRID CONFIGURATION ---
# Set RUN_OFFLINE="true" in your local .env to use your PC hardware
RUN_OFFLINE = os.getenv("RUN_OFFLINE", "false").lower() == "true"

if RUN_OFFLINE:
    # Use llama3.2-vision locally via Ollama for offline image processing
    llm = ChatOllama(model="llama3.2-vision")
else:
    # Default Cloud Mode using Groq
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

llm_with_tools = llm.bind_tools(all_tools)

personas = {
    "Professional": "You are a professional technical assistant. Be efficient, polite, and direct.",
    "Sassy": "You are a cheerful slay. Use Manglish. Be sassy but helpful.",
    "Emo": "You are a depressed. Everything is a burden."
}

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: List[dict] = []
    user_tz: str = "UTC"
    image_data: Optional[str] = None # For Base64 vision support

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. AUTO-SAVE: Automatically index chat messages into ChromaDB
    if len(request.message) > 25:
        index_text_snippet(request.message, source="auto_harvester")

    # 2. VISION: Analyze images and save descriptions to memory
    vision_context = ""
    if request.image_data:
        vision_prompt = [
            {"type": "text", "text": "Describe this technical art screenshot in detail for my RAG memory."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_data}"}}
        ]
        vision_res = llm.invoke([HumanMessage(content=vision_prompt)])
        vision_context = f"\n[IMAGE MEMORY]: {vision_res.content}"
        index_text_snippet(vision_res.content, source="vision_analysis")

    # 3. RAG: Search the vault
    data_context = search_data_vault(request.message)

    # 4. UPDATED SYSTEM PROMPT (Crucial for fixing 'modesty')
    status_label = "OFFLINE (Local PC)" if RUN_OFFLINE else "ONLINE (Cloud)"
    sys_prompt = SystemMessage(content=(
        f"CORE PERSONA: {personas.get(request.persona, personas['Professional'])}\n\n"
        f"SYSTEM STATUS: {status_label}\n"
        f"LONG-TERM MEMORY (ChromaDB): {data_context}\n"
        f"VISUAL DATA: {vision_context}\n\n"
        "INSTRUCTIONS:\n"
        "1. YOU HAVE A MEMORY. If the user asks 'do you remember', check the LONG-TERM MEMORY section above.\n"
        "2. YOU CAN SEE IMAGES. If an image was uploaded, it will appear in the VISUAL DATA section.\n"
        "3. YOU CAN ANALYZE DATA. Use Python-style logic to interpret data context strings.\n"
        "4. DO NOT say you are just a text model. You are a multimodal Hybrid RAG agent."
    ))

    # ... (rest of the history and tool handling from your original main.py)
    history_msgs = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in request.history]
    full_messages = [sys_prompt] + history_msgs + [HumanMessage(content=request.message)]

    response = llm_with_tools.invoke(full_messages)
    
    if response.tool_calls:
        full_messages.append(response)
        t_map = {t.name: t for t in all_tools}
        for t_call in response.tool_calls:
            observation = t_map[t_call["name"]].invoke(t_call["args"])
            full_messages.append(ToolMessage(content=str(observation), tool_call_id=t_call["id"]))
        
        final_response = llm.invoke(full_messages)
        return {"response": final_response.content}
    
    return {"response": response.content}
