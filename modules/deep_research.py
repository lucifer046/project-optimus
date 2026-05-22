# ┌────────────────────────────────────────────────────────────────────────┐
# │                            deep_research.py                            │
# │                Autonomous Deep Synthesis Research Engine               │
# └────────────────────────────────────────────────────────────────────────┘
"""
This module implements an autonomous, multi-step deep research agent.
It takes a broad topic, decomposes it into highly specific sub-queries,
gathers context blocks from the web via DuckDuckGo, and synthesizes 
an exhaustive technical whitepaper in Markdown format.
"""

import os
import re
from datetime import datetime
from dotenv import dotenv_values

# Fallback import handles modern ddgs vs legacy duckduckgo_search libraries
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Robust relative path imports across standalone and package execution
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

# Resolve absolute pathways to dynamically locate .env from any execution directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if __file__ else "."
env_vars = dotenv_values(os.path.join(root, ".env")) or {}

# Number of parallel sub-questions/search angles to split the research topic into
MAX_SUB_QUESTIONS = int(env_vars.get("MAX_SUB_QUESTIONS", "3"))

# Initialize centralized intelligence switcher matrix
engine = CentralizedLLMEngine()

# Ensure dedicated Reports output directory exists
REPORTS_DIR = "Reports"
if not os.path.exists(REPORTS_DIR):
    os.mkdir(REPORTS_DIR)

# ┌────────────────────────────────────────────────────────────────────────┐
# │                             AGENT TOOLS                                │
# └────────────────────────────────────────────────────────────────────────┘

def generate_sub_queries(topic):
    """
    Decomposes a broad target topic into specific, optimized search queries using the LLM.

    Parameters:
        topic (str): The broad central research topic.

    Returns:
        list: A list of specific search query strings.
    """
    print_info(f"Decomposing core topic: '{topic}'...")
    
    # Prompt asks the model to function strictly as a query planner.
    # It forces a comma-separated output for easy structural parsing in python.
    prompt = f"""
    You are an expert research planner. Break down the following topic into {MAX_SUB_QUESTIONS} highly specific, distinct Google search queries that will yield the most comprehensive information.
    Topic: {topic}
    Output ONLY a comma-separated list of {MAX_SUB_QUESTIONS} queries. No extra text, bullet points, or numbering.
    """

    api_messages = [{"role": "user", "content": prompt}]
    
    response_text = ""
    # Silently accumulate stream generator chunks for program parsing
    for chunk in engine.generate_chat_stream(api_messages):
        response_text += chunk
        
    # Split by comma, clean whitespace, and limit results to MAX_SUB_QUESTIONS
    queries = [q.strip() for q in response_text.split(',')]
    return queries[:MAX_SUB_QUESTIONS]

def bulk_scrape(query):
    """
    Executes a web search on DuckDuckGo and compiles the results into a context block.

    Parameters:
        query (str): The search query vector to process.

    Returns:
        str: A formatted context block of titles and body text.
    """
    try:
        # Using DDGS context manager guarantees socket resource cleanup
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            
        if not results:
            return ""

        # Format raw search items into clear structured contextual data blocks
        compiled_text = f"\n--- Search Topic: {query} ---\n"
        for result in results:
            title = result.get('title', 'Unknown')
            body = result.get('body', '')
            compiled_text += f"[{title}]: {body}\n"
            
        return compiled_text
    except Exception as e:
        print_warning(f"Data retrieval failed for '{query}': {e}")
        return ""

# ┌────────────────────────────────────────────────────────────────────────┐
# │                             MAIN ENGINE                                │
# └────────────────────────────────────────────────────────────────────────┘

def DeepResearchEngine(topic):
    """
    Orchestrates the multi-phase deep research loop:
    1. Topic Decomposition: Generates structured query sub-vectors.
    2. Extraction Loop: Scrapes each sub-vector to compile a comprehensive raw context pool.
    3. Synthesis Phase: Instructs the LLM to write a professional Markdown report.
    4. Export Phase: Saves the generated synthesis to a timestamped Markdown file.
    """
    try:
        console.print("\n[bold magenta]=== INITIATING DEEP RESEARCH PROTOCOL ===[/bold magenta]")
        
        # Step 1: Breakdown broad topic into granular components
        sub_queries = generate_sub_queries(topic)
        print_success(f"Generated research vectors: {sub_queries}")
        
        # Step 2: Accumulate scraped context for each query vector
        master_context = ""
        for i, q in enumerate(sub_queries, 1):
            print_info(f"Scraping Vector [{i}/{len(sub_queries)}]: {q}...")
            master_context += bulk_scrape(q)
            
        if not master_context.strip():
            print_error("Failed to retrieve sufficient data blocks. Aborting.")
            return
            
        # Step 3: Prompt LLM to synthesize raw context into a formal whitepaper
        print_system("Data pool compiled. Synthesizing technical whitepaper...")
        
        synthesis_prompt = f"""
        You are a world-class research analyst. Write an exhaustive, highly structured technical report on the following topic based ONLY on the provided context data.
        
        Topic: {topic}
        
        Requirements:
        - Use professional Markdown formatting (Headers, Bullet points, Bold text).
        - Structure it logically: Executive Summary, Deep Dive (by categories), and Conclusion.
        - Do not use conversational filler (e.g., "Here is your report").
        - If the context contradicts itself, mention the discrepancy.
        
        [RAW SCRAPED CONTEXT]
        {master_context}
        """
        
        api_messages = [{"role": "user", "content": synthesis_prompt}]
        
        # Stream the report text live on the terminal for a premium user experience
        final_report = ""
        console.print("\n[bold white]Generating Document Live:[/bold white]\n", style="dim")
        
        for chunk in engine.generate_chat_stream(api_messages):
            console.print(chunk, end="", style="cyan")
            final_report += chunk
            
        console.print("\n")
        
        # Step 4: Export report to local disk
        # Clean topic text to build a safe filename, strip special characters and restrict size
        clean_topic = re.sub(r'[^a-zA-Z0-9_\- ]', '', topic)
        clean_topic = clean_topic.replace(" ", "_")[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{clean_topic}_{timestamp}.md"
        filepath = os.path.join(REPORTS_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(final_report)
            
        print_success(f"Research protocol complete. Document saved to: {filepath}")
        
    except Exception as e:
        print_error(f"Deep Research fatal failure: {e}")

# ┌────────────────────────────────────────────────────────────────────────┐
# │                         DIAGNOSTIC TEST NODE                           │
# └────────────────────────────────────────────────────────────────────────┘

if __name__ == "__main__":
    # Test entrypoint to execute research tasks locally in an interactive terminal sandbox
    print_banner("KAYRA RESEARCH ENGINE", "Autonomous Deep Synthesis Sandbox")
    
    while True:
        try:
            user_topic = console.input("\n[bold cyan]Target Topic >[/bold cyan] ").strip()
            
            if not user_topic:
                continue
                
            if user_topic.lower() in ["exit", "quit", "bye"]:
                print_system("Terminating research workspace.")
                break
                
            DeepResearchEngine(user_topic)
            
        except KeyboardInterrupt:
            console.print()
            print_system("Manual interrupt. Shutting down research agent.")
            break