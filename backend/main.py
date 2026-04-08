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

app = FastAPI(title="Global Vision AI: Hybrid Prototype")

# --- HYBRID CONFIGURATION ---
# Set RUN_OFFLINE="true" in your local .env to use your PC's GPU/CPU
RUN_OFFLINE = os.getenv("RUN_OFFLINE", "false").lower() == "true"

if RUN_OFFLINE:
    # LOCAL MODE: Uses your PC hardware (Requires Ollama installed)
    # We use llama3.2-vision because it can handle the image analysis locally
    llm = ChatOllama(model="llama3.2-vision")
else:
    # CLOUD MODE: Uses Groq
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
    image_data: Optional[str] = None # For Base64 images from Streamlit

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    result = index_any_csv(content, file.filename)
    return result

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. AUTO-SAVE HARVESTER: Save substantial technical chat messages
    if len(request.message) > 30:
        index_text_snippet(request.message, source="auto_harvester")

    # 2. VISION ANALYSIS: If an image is sent, analyze it and save the description
    vision_context = ""
    if request.image_data:
        vision_prompt = [
            {"type": "text", "text": "Describe this technical art screenshot in detail for my long-term memory."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_data}"}}
        ]
        vision_res = llm.invoke([HumanMessage(content=vision_prompt)])
        vision_context = f"\n[CURRENT VISUAL CONTEXT]: {vision_res.content}"
        # Save description so the agent 'remembers' the image tomorrow
        index_text_snippet(vision_res.content, source="vision_analysis")

    # 3. RAG RETRIEVAL: Pull context from ChromaDB (Works offline too!)
    data_context = search_data_vault(request.message)

    # 4. HYBRID SYSTEM PROMPT
    status_label = "OFFLINE (Local PC)" if RUN_OFFLINE else "ONLINE (Cloud)"
    sys_prompt = SystemMessage(content=(
        f"CORE PERSONA: {personas.get(request.persona, personas['Professional'])}\n\n"
        f"SYSTEM STATUS: {status_label}\n"
        f"LONG-TERM MEMORY: {data_context}\n"
        f"VISUAL CONTEXT: {vision_context}\n\n"
        f"REFERENCE - User's Home Timezone: {request.user_tz}\n"
        "INSTRUCTIONS:\n"
        "1. You have a long-term memory vault. Use the LONG-TERM MEMORY section to answer 'Do you remember' questions.\n"
        "2. If offline, gracefully explain that web-dependent tools (Search/Wiki) might be restricted.\n"
        "3. Stay in character consistently."
    ))

    history_msgs = []
    for m in request.history:
        if m["role"] == "user": 
            history_msgs.append(HumanMessage(content=m["content"]))
        else: 
            history_msgs.append(AIMessage(content=m["content"]))

    full_messages = [sys_prompt] + history_msgs + [HumanMessage(content=request.message)]

    # 5. EXECUTION & TOOL CALLING
    response = llm_with_tools.invoke(full_messages)
    
    if response.tool_calls:
        full_messages.append(response)
        t_map = {t.name: t for t in all_tools}
        for t_call in response.tool_calls:
            # Handle Web Tools if Offline
            if RUN_OFFLINE and t_call["name"] in ["fact_check_search", "wikipedia"]:
                observation = "System: Cannot search web while in Offline Mode."
            else:
                observation = t_map[t_call["name"]].invoke(t_call["args"])
            full_messages.append(ToolMessage(content=str(observation), tool_call_id=t_call["id"]))
        
        final_response = llm.invoke(full_messages)
        return {"response": final_response.content}
    
    return {"response": response.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
