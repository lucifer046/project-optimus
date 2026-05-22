# ┌────────────────────────────────────────────────────────────────────────┐
# │                          real_time_search.py                           │
# │     Real-Time Search & Retrieval-Augmented Generation (RAG) Engine     │
# └────────────────────────────────────────────────────────────────────────┘
"""
This module implements the real-time search subsystem for the KAYRA assistant.
It performs web searches via DuckDuckGo (using auto-switching HTML/text backends),
augments the LLM system prompt with live retrieved document contexts, and 
orchestrates conversational streaming with support for dual-tier memory management.
"""

import os 
import datetime
import json
import shutil
from dotenv import dotenv_values

# Fallback block to safely import DDGS from ddgs or duckduckgo_search packages
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Robust relative path imports supporting standalone and package-level execution
try:
    from .llm_engine import CentralizedLLMEngine
except ImportError:
    try:
        from modules.llm_engine import CentralizedLLMEngine
    except ImportError:
        from llm_engine import CentralizedLLMEngine

try:
    from .utils import print_banner, print_info, print_success, print_warning, print_error, print_system, console
except ImportError:
    try:
        from modules.utils import print_banner, print_info, print_success, print_warning, print_error, print_system, console
    except ImportError:
        from utils import print_banner, print_info, print_success, print_warning, print_error, print_system, console

# ┌────────────────────────────────────────────────────────────────────────┐
# │                            CONFIGURATION                               │
# └────────────────────────────────────────────────────────────────────────┘

# Dynamically calculate project root directory to ensure .env is discovered reliably
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if __file__ else "."
env_vars = dotenv_values(os.path.join(root, ".env")) or {}

assistant_name = env_vars.get("ASSISTANT_NAME", "").strip()
if not assistant_name:
    assistant_name = "Kayra"

# Centralized LLM orchestrator handling model routing and streaming
engine = CentralizedLLMEngine()

# Persistent conversation history storage files
DB_FILE = r"data\conversation.json"
BACKUP_FILE = r"data\conversation_backup.json"

# ┌────────────────────────────────────────────────────────────────────────┐
# │                         WEB SEARCH SUBSYSTEM                           │
# └────────────────────────────────────────────────────────────────────────┘

def WebSearch(query):
    """
    Executes a zero-cost, unauthenticated search query on DuckDuckGo.
    Extracts and structures the top 5 relevant documents as live context.

    Parameters:
        query (str): The search query phrase.

    Returns:
        str: A formatted block containing titles, content snippets, and URLs.
             Returns None if search results are completely empty or fail.
    """
    try:
        # DDGS context manager guarantees TCP/HTTP socket cleanup on termination
        with DDGS() as ddgs:
            # DuckDuckGo's standard API ('auto' backend) can hit rate limits or block empty search strings.
            # We execute the default backend search first.
            results = list(ddgs.text(query, max_results=5))
            
            # If rate-limited or blocked (returning empty), fallback to the scraper HTML backend
            if not results:
                results = list(ddgs.text(query, backend="html", max_results=5))

        if not results:
            return None

        # Build a highly structured context template to ground the LLM's response
        formatted_results = f"The live web search results for '{query}' are:\n[START OF SEARCH DATA]\n"
        for idx, result in enumerate(results, 1):
            title = result.get("title", "Untitled Document")
            body = result.get("body", 'No snippet text available.')
            link = result.get('href', 'No link available.')
            formatted_results += f"Document [{idx}]:\nTitle: {title}\nContent: {body}\nLink: {link}\n\n"
        formatted_results += "[END OF SEARCH DATA]"

        return formatted_results
    except Exception as e:
        print_warning(f"Web scraping pipeline execution failed: {e}")
        return None
    
# ┌────────────────────────────────────────────────────────────────────────┐
# │                       MEMORY MANAGEMENT LAYER                          │
# └────────────────────────────────────────────────────────────────────────┘

def AnswerModifier(Answer):
    """
    Cleans structural whitespace anomalies to save canvas tokens and maximize terminal layout density.

    Parameters:
        Answer (str): Raw LLM response string.

    Returns:
        str: Cleaned response with empty lines removed.
    """
    lines = Answer.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def load_memory():
    """
    Loads persistent conversation data from the local JSON database file.
    
    Fault-Tolerance:
        If the primary JSON database is corrupted or missing due to a sudden program halt,
        it intercepts the exception and attempts to load from the secondary rolling backup.
    """
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        try:
            with open(BACKUP_FILE, "r") as f:
                data = json.load(f)
            print_warning("Primary index compromised. Restored data from rolling backup.")
            return data
        except:
            return []

def save_memory(memory_list):
    """
    Saves the conversation database to disk using an atomic transactional pattern.
    
    Corrupt-Prevention Strategy:
        1. Write content to the secondary backup file first.
        2. Once written successfully, copy the backup to the primary DB file.
        This ensures that if a crash occurs mid-write, the primary database remains completely uncorrupted.
    """
    try:
        with open(BACKUP_FILE, "w") as f:
            json.dump(memory_list, f, indent=4)
        shutil.copy(BACKUP_FILE, DB_FILE)
        return True
    except Exception as e:
        print_error(f"Persistent storage transaction failed: {e}")
        return False

# ┌────────────────────────────────────────────────────────────────────────┐
# │                     MEMORY SPACE INITIALIZATION                        │
# └────────────────────────────────────────────────────────────────────────┘

# Long-term persistent memory loaded from database
permanent_memory = load_memory()
# Short-term volatile RAM cache serving as a sliding session context
session_memory = []

def RealTimeInformation():
    """
    Compiles chronological host parameters to anchor the LLM in real-world time.
    """
    current_date_time = datetime.datetime.now()
    data = f"Current Time: {current_date_time.strftime('%I:%M %p')}\n"
    data += f"Day: {current_date_time.strftime('%A')}, Date: {current_date_time.strftime('%d %B %Y')}"
    return data

# ┌────────────────────────────────────────────────────────────────────────┐
# │                       MAIN INTERACTION PIPELINE                        │
# └────────────────────────────────────────────────────────────────────────┘

def RealTimeSearchEngine(query):
    """
    Orchestrates the live Web Search RAG loop:
    1. Dispatches search terms to the unauthenticated DuckDuckGo scraper.
    2. Constructs a temporary system context template injected with retrieved documentation.
    3. Merges long-term persistent registers and short-term sliding context buffers.
    4. Handles real-time live streaming of answers on stdout.
    5. Saves conversational frames to sliding memory and long-term stores (if triggered).
    """
    global permanent_memory, session_memory

    try:
        # Step 1: Query the Web Scraper
        print_info(f"Polling active web networks: '{query}...")
        search_context = WebSearch(query)

        # Step 2: Inject search results into the prompt context payload
        if search_context:
            prompt_payload = f"""
            [Context from Live Web Search]
            {search_context}

            [User Query]
            Based ONLY on the web search context provided above, answer the following query accurately:
            {query}
            If the search context does not contain the full or complete information needed to answer the query, 
            please supplement it with your own accurate pre-trained knowledge to provide a complete response.
            """
        else:
            print_warning("Network returned empty search tokens. Reverting to frozen parameters.")
            prompt_payload = query

        # Step 3: Compile System Prompt and Temporal Calibration data
        identity_prompt = engine.get_identity_prompt()

        api_messages = [
            {"role": "system", "content": identity_prompt + "\n\n" + RealTimeInformation()}
        ]

        # Step 4: Merge permanent conversation records
        if len(permanent_memory) > 0:
            for msg in permanent_memory:
                api_messages.append(msg)
        
        # Step 5: Merge short-term volatile RAM sliding context window (limited to last 6 entries)
        recent_session = session_memory[-6:]
        for msg in recent_session:
            api_messages.append(msg)
        
        # Step 6: Append the compiled injected payload
        api_messages.append({"role": "user", "content": prompt_payload})

        response_text = ""
        console.print("\n[bold white]Streaming Real-Time Response Live:[/bold white] ", end="")

        # Step 7: Stream generated response token-by-token
        for chunk in engine.generate_chat_stream(api_messages):
            console.print(chunk, end="", style="italic green")
            response_text += chunk
        
        console.print("\n")

        # Step 8: Clean and format final text
        response_text = AnswerModifier(response_text)

        # Step 9: Store original query (without scraped noise) and output to session RAM
        session_memory.append({"role": "user", "content": query})
        session_memory.append({"role": "assistant", "content": response_text})

        # Step 10: Scan for explicit persistent indexing commands
        triggers = ["store this", "remember this", "save this", "memorize this", "note this"]
        if any(trigger in query.lower() for trigger in triggers):
            permanent_memory.append({"role": "user", "content": query})
            permanent_memory.append({"role": "assistant", "content": response_text})

            if save_memory(permanent_memory):
                print_success(f"Secure context verified. Stored in {assistant_name} structural database.")

        return response_text

    except Exception as e:
        print_error(f"Pipeline failure inside Real-Time search node execution: {e}")
        return ""

# ┌────────────────────────────────────────────────────────────────────────┐
# │                     DIAGNOSTIC TEST RUNTIME BLOCK                      │
# └────────────────────────────────────────────────────────────────────────┘

if __name__ == "__main__":
    # Test playground enabling real-time search evaluation and RAG synthesis locally
    print_banner("KAYRA REALTIME SEARCH ENGINE", f"Live RAG Web Synthesis Sandbox")
    print_success(f"Realtime Subsystem Node Online. Free Web Scraping Interface Active.")

    while True:
        try:
            user_query = console.input("[bold cyan]Search >[/bold cyan] ").strip()

            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit", "bye"]:
                print_system("Terminating real-time tracking workspace instance.")
                break

            RealTimeSearchEngine(user_query)

        except KeyboardInterrupt:
            console.print()
            print_system("Terminal Halt Event Intercepted. Shutting down system search.")
            break