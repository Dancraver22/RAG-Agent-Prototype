import os
import pytz
from datetime import datetime
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from tavily import TavilyClient

# --- 1. SEARCH TOOL ---
@tool
def fact_check_search(query: str):
    """
    REQUIRED for any factual questions. 
    Returns the most up-to-date verified facts from the live web.
    """
    try:
        # Use os.getenv for Render/Production safety
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        
        results = client.search(query=query, search_depth="advanced", max_results=5, include_answer=True)
        
        direct_answer = results.get('answer', "")
        sources = [f"SOURCE DATA: {r['content']} (Link: {r['url']})" for r in results['results']]
        context = "\n---\n".join(sources)
        
        return f"VERIFIED ANSWER: {direct_answer}\n\nSUPPORTING EVIDENCE:\n{context}"
    except Exception as e:
        return f"Search Error: {str(e)}"

# --- 2. DYNAMIC TIME CONVERTER ---
@tool
def get_world_clock(location: str):
    """
    REQUIRED whenever the user asks 'what time is it'.
    Use the user's home location if they don't mention a city.
    """
    try:
        search_term = location.strip().replace(" ", "_").lower()
        best_match = next((tz for tz in pytz.all_timezones if search_term in tz.lower()), None)
        
        if not best_match:
            return f"Timezone for {location} not found."

        target_tz = pytz.timezone(best_match)
        now = datetime.now(target_tz)
        return f"The current time in {location} is {now.strftime('%I:%M %p')}."
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. WIKIPEDIA ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- 4. ARCHIVE TOOL ---
@tool
def save_research_to_file(data: str):
    """Saves text findings into a local .txt file."""
    try:
        filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(data)
        return f"✅ Archived to {filename}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

all_tools = [get_world_clock, fact_check_search, save_research_to_file, wiki_tool]
