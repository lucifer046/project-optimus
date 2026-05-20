import os
import sys
import time

# Reconfigure stdout/stderr to support UTF-8 characters on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Append the project root folder to the Python path to allow importing the modules package
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from modules.llm_engine import CentralizedLLMEngine

def test_run():
    print("=============================================")
    print("STARTING LLM ENGINE INTEGRATION TEST")
    print("=============================================\n")

    # 1. Initialize the Engine
    try:
        engine = CentralizedLLMEngine()
    except Exception as e:
        print(f"[CRITICAL] Failed to initialize engine instance: {e}")
        return

    print(f"\n[Active Target Setup] Network Status -> Is Online Mode? {engine.is_online}")
    print("---------------------------------------------")

    # 2. Test Layer 1: Decision Making Model (DMM / Classification)
    print("\n[TEST 1/2] Testing Intent Classification (DMM)...")
    test_queries = [
        "What is the weather like in New Delhi today?",
        "Can you help me write a quick python sorting algorithm?",
        "Run a deep research pass on sodium ion battery scalability"
    ]

    for q in test_queries:
        print(f"\nSending Query: '{q}'")
        start = time.time()
        try:
            intent_result = engine.classify_intent(q)
            elapsed = time.time() - start
            print(f"DMM Response Array: {intent_result} (Took {elapsed:.2f}s)")
        except Exception as e:
            print(f"[Error in DMM Phase]: {e}")

    print("\n---------------------------------------------")

    # 3. Test Layer 2: Chat Stream (Token Generation)
    print("\n[TEST 2/2] Testing Chat Streaming Generation...")
    
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep your response under 15 words."},
        {"role": "user", "content": "Say hello and give me one word of motivation."}
    ]

    print("\nStreaming Output Live: ", end="", flush=True)
    start_stream = time.time()
    chunks_received = 0
    
    try:
        # Loop through the stream generator
        for chunk in engine.generate_chat_stream(test_messages):
            print(chunk, end="", flush=True)
            chunks_received += 1
        
        elapsed_stream = time.time() - start_stream
        print(f"\n\nStream Finished successfully! Received {chunks_received} chunks in {elapsed_stream:.2f}s")
        
    except Exception as e:
        print(f"\n[Error in Streaming Phase]: {e}")

    print("\n=============================================")
    print("TEST NODE RUN COMPLETE")
    print("=============================================")

if __name__ == "__main__":
    test_run()