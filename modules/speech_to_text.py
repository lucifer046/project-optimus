# ┌────────────────────────────────────────────────────────────────────────┐
# │                           speech_to_text.py                            │
# │                  Continuous Background STT Engine                      │
# └────────────────────────────────────────────────────────────────────────┘
"""
This module implements a continuous, background Speech-to-Text (STT) transcription system
using the HTML5 Web Speech API running within a headless Selenium-controlled Chrome instance.
It employs voice activity detection (VAD) to segment spoken audio into discrete sentences
without losing words during processing delays.
"""

import os
import re
import time
import urllib.parse
import atexit
import mtranslate as mt
from dotenv import dotenv_values
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Robust imports supporting relative paths across all execution contexts
try:
    from .utils import print_info, print_warning, print_error, print_system, print_success, print_banner, console
except ImportError:
    try:
        from modules.utils import print_info, print_warning, print_error, print_system, print_success, print_banner, console
    except ImportError:
        from utils import print_info, print_warning, print_error, print_system, print_success, print_banner, console

# ┌────────────────────────────────────────────────────────────────────────┐
# │        IN-BROWSER WEB SPEECH API & VAD SILENCE QUEUING HTML/JS         │
# └────────────────────────────────────────────────────────────────────────┘
# We run this minimal web page inside our headless browser session.
# It configures the Web Speech API (webkitSpeechRecognition) and implements real-time silence detection:
# - Continually listens for speech input.
# - If silence is detected for more than `silenceLimit` ms, the accumulated interim text buffer is finalized.
# - This finalized sentence is appended to a global JS list `window.speechQueue`.
# - In case of browser engine interruptions or pauses, it automatically restarts without losing context.
html_code = """<!DOCTYPE html>
<html lang="en">
<head>
    <title>Speech Recognition</title>
</head>
<body>
    <p id="status">idle</p>
    <script>
        let recognition;
        let lastResultTime = 0;
        let isSpeaking = false;
        let silenceLimit = 800; // default 800ms silence gap
        let checkInterval;
        
        // Asynchronous queue to hold finalized sentences
        window.speechQueue = [];
        let currentText = "";

        const statusEl = document.getElementById('status');

        function startContinuousRecognition(lang, silenceMs) {
            silenceLimit = silenceMs || 800;
            window.speechQueue = [];
            currentText = "";
            isSpeaking = false;
            lastResultTime = Date.now();
            statusEl.textContent = "listening";

            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = lang || 'en-US';
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onstart = () => {
                statusEl.textContent = "listening";
            };

            recognition.onspeechstart = () => {
                isSpeaking = true;
                statusEl.textContent = "speaking";
                lastResultTime = Date.now();
            };

            recognition.onresult = (event) => {
                isSpeaking = true;
                statusEl.textContent = "speaking";
                lastResultTime = Date.now();

                let finalTranscript = "";
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript + " ";
                    }
                }
                if (finalTranscript) {
                    currentText += finalTranscript;
                }
            };

            recognition.onerror = (event) => {
                // If it hits standard network/media interrupts, auto-restart
                if (event.error === 'network' || event.error === 'aborted') {
                    restartRecognition();
                } else {
                    statusEl.textContent = "error: " + event.error;
                }
            };

            recognition.onend = () => {
                // Restart continuously if stopped by Chrome system
                if (statusEl.textContent !== "stopped") {
                    restartRecognition();
                }
            };

            recognition.start();

            // Check for silence gap every 100ms
            if (checkInterval) clearInterval(checkInterval);
            checkInterval = setInterval(() => {
                if (isSpeaking && (Date.now() - lastResultTime > silenceLimit)) {
                    let completedSentence = currentText.trim();
                    if (completedSentence) {
                        window.speechQueue.push(completedSentence);
                        currentText = ""; // Clear buffer for next sentence
                    }
                    isSpeaking = false;
                    statusEl.textContent = "listening";
                }
            }, 100);
        }

        function restartRecognition() {
            if (recognition) {
                try { recognition.stop(); } catch(e) {}
            }
            setTimeout(() => {
                try { recognition.start(); } catch(e) {}
            }, 50);
        }

        function stopContinuousRecognition() {
            statusEl.textContent = "stopped";
            clearInterval(checkInterval);
            if (recognition) {
                recognition.onend = null;
                recognition.stop();
            }
        }
    </script>
</body>
</html>"""


class SpeechToTextEngine:
    """
    Continuous Asynchronous Speech-to-Text Engine.
    Orchestrates the headless Selenium Chrome instance, VAD queuing, and translation threads.
    """
    def __init__(self, language=None, silence_limit=0.8):
        """
        Initializes the persistent Chrome browser and activates continuous recording.
        
        Parameters:
            language (str): Target language code (e.g. 'en-US', 'hi-IN'). Defaults to INPUT_LANGUAGE in .env.
            silence_limit (float): Silence detection threshold in seconds (VAD gap size).
        """
        # Load language configurations from environmental setups
        if not language:
            env_vars = dotenv_values(".env") or {}
            language = env_vars.get("INPUT_LANGUAGE", "en-US")

        self.language = language
        self.silence_limit_ms = int(silence_limit * 1000)

        # Set Chrome Options to allow microphone access without UI authorization prompts
        chrome_options = Options()
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.3"
        
        # Headless Configuration
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument(f"user-agent={user_agent}")
        chrome_options.add_argument("--use-fake-ui-for-media-stream") # Bypasses browser mic permission popup
        
        # Advanced Headless Performance Optimizations (Fast boot, low RAM, zero GPU compile delays)
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        chrome_options.add_argument("--disable-background-networking")
        chrome_options.add_argument("--disable-sync")

        print_system("Booting headless Chrome browser session for background STT...")
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Inject the HTML code as an inline Data URL to eliminate disk pollution/temp files
        data_url = "data:text/html;charset=utf-8," + urllib.parse.quote(html_code)
        self.driver.get(data_url)
        print_success("Headless Chrome background session active & ready.")

        # Resolve path to the centralized LOWERCASE 'data/Files' directory relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.temp_dir_path = os.path.join(project_root, "data", "Files")
        os.makedirs(self.temp_dir_path, exist_ok=True)

        # Start continuous audio capture recognition instantly inside the web page context
        self.driver.execute_script(
            "startContinuousRecognition(arguments[0], arguments[1]);",
            self.language,
            self.silence_limit_ms
        )
        print_info("Continuous speech voice activity detection (VAD) monitor active.")

        # Register robust process termination cleanup handler
        atexit.register(self.shutdown)

    def set_assistant_status(self, status):
        """
        Updates the status file so external applications/GUIs can show 'Listening...' or errors.
        
        Parameters:
            status (str): Current state description text.
        """
        try:
            with open(rf"{self.temp_dir_path}/status.data", "w", encoding="utf-8") as f:
                f.write(status)
        except Exception:
            pass

    def clear_queue(self):
        """Flushes any accumulated speech in the background browser queue."""
        try:
            if self.driver:
                self.driver.execute_script("window.speechQueue = [];")
        except Exception:
            pass

    def listen_and_transcribe(self):
        """
        Blocks the Python main thread until a completed sentence is finalized in Chrome's queue,
        then pops, translates to English (if non-English), formats, and returns the result.
        
        Returns:
            str: The capitalized, formatted English query transcript.
        """
        self.set_assistant_status("Listening...")
        
        try:
            while True:
                if not self.driver:
                    return None

                # Fetch and remove (pop) the oldest sentence entry from the JS-side queue array
                text = self.driver.execute_script("return window.speechQueue.shift();")

                if text:
                    self.set_assistant_status("Translating...")
                    translated_text = translate_query(text)
                    formatted_text = format_query(translated_text)
                    return formatted_text
                
                # Check for critical runtime errors reported inside Chrome engine
                status = self.driver.find_element(By.ID, "status").text
                if status.startswith("error:"):
                    error_msg = status.replace("error: ", "")
                    print_error(f"Chrome STT Internal Error: {error_msg}")
                    return ""
                
                # Super-low CPU polling sleep interval (50ms) to ensure minimal host thread impact
                time.sleep(0.05)

        except KeyboardInterrupt:
            return None

    def clear_queue(self):
        """Purges the JavaScript speech queue. Used to prevent TTS echoing loops."""
        if self.driver:
            try:
                self.driver.execute_script("window.speechQueue = [];")
            except BaseException:
                pass

    def shutdown(self):
        """Closes the Selenium webdriver and terminates Chrome cleanly to prevent zombie processes."""
        try:
            atexit.unregister(self.shutdown)
        except Exception:
            pass

        if hasattr(self, 'driver') and self.driver:
            print_system("Shutting down headless Chrome background session...")
            try:
                self.driver.execute_script("stopContinuousRecognition();")
            except BaseException:
                pass
            try:
                self.driver.quit()
            except BaseException:
                pass
            self.driver = None


# ┌────────────────────────────────────────────────────────────────────────┐
# │                  FORMATTING & TRANSLATION UTILITIES                    │
# └────────────────────────────────────────────────────────────────────────┘

def format_query(query):
    """
    Cleans and structures raw synthesized query speech:
    - Normalizes word boundaries.
    - Resolves typical query interrogators (e.g. what, where, can you) to append a question mark '?'.
    - Appends periods '.' to generic command declarations.
    - Capitalizes the final text string for premium presentation.
    
    Parameters:
        query (str): The raw text sequence to format.
        
    Returns:
        str: The structured, formatted transcript.
    """
    new_query = query.lower().strip()
    query_words = new_query.split()
    question_words = [
        "what", "where", "when", "why", "how", "who", "which", "whom", "whose", "whatsoever", "wherever", 
        "whenever", "whichever", "can you", "what's", "where's", "when's", "why's", "how's", 
        "who's", "which's", "whom's", "whose's"
    ]

    if not query_words: 
        return ""
    
    # Interrogate first word or interior structures for questioning contexts
    if any(word + " " in new_query for word in question_words) or query_words[0] in question_words:
        if new_query[-1] in ['.', '?', '!']:
            new_query = new_query[:-1] + "?"
        else:
            new_query += "?"
    else:
        if new_query[-1] in ['.', '?', '!']:
            new_query = new_query[:-1] + "."
        else:
            new_query += "."
    return new_query.capitalize()


def translate_query(query):
    """
    Translates non-English input speech into English text using mtranslate.
    
    Parameters:
        query (str): The input text in any foreign tongue.
        
    Returns:
        str: The translated English equivalent in capitalized format.
    """
    # Pre-translation phonetic corrections:
    # Google Speech-to-Text in Hindi ('hi-IN') transcribes the phonetic name "Kayra"
    # either as the real Hindi name "कायरा" or the homophonic "कायर" (meaning "coward").
    # We swap both to "Kayra" before translating so they remain stable.
    corrected_query = query
    if "कायर" in corrected_query:
        corrected_query = corrected_query.replace("कायर", "Kayra")
    if "कायरा" in corrected_query:
        corrected_query = corrected_query.replace("कायरा", "Kayra")
        
    english_query = mt.translate(corrected_query, "en", "auto")
    
    # Post-translation robustness:
    # Handle any cases where English/Hinglish transcribes "kaira" or "coward".
    # We perform case-insensitive whole-word replacements to enforce the "Kayra" spelling.
    english_query = re.sub(r"\bcowards\b", "Kayras", english_query, flags=re.IGNORECASE)
    english_query = re.sub(r"\bcoward's\b", "Kayra's", english_query, flags=re.IGNORECASE)
    english_query = re.sub(r"\bcoward\b", "Kayra", english_query, flags=re.IGNORECASE)
    english_query = re.sub(r"\bkairas\b", "Kayras", english_query, flags=re.IGNORECASE)
    english_query = re.sub(r"\bkaira's\b", "Kayra's", english_query, flags=re.IGNORECASE)
    english_query = re.sub(r"\bkaira\b", "Kayra", english_query, flags=re.IGNORECASE)
    
    return english_query.capitalize()


# ┌────────────────────────────────────────────────────────────────────────┐
# │                 BACKWARD COMPATIBILITY CLASS ALIASES                   │
# └────────────────────────────────────────────────────────────────────────┘
OnlineSpeechEngine = SpeechToTextEngine
SetAssistantStatus = SpeechToTextEngine.set_assistant_status
QueryModifier = format_query
UniversalTranslator = translate_query

_legacy_engine = None

def recognize_speech():
    """
    Legacy wrapper function to maintain backwards-compatibility.
    Automatically initializes a singleton SpeechToTextEngine session on demand.
    """
    global _legacy_engine
    if _legacy_engine is None:
        _legacy_engine = SpeechToTextEngine()
    
    return _legacy_engine.listen_and_transcribe()

SpeechRecognition = recognize_speech


# ┌────────────────────────────────────────────────────────────────────────┐
# │                     MAIN SCRIPT TEST ENTRYPOINT                        │
# └────────────────────────────────────────────────────────────────────────┘
if __name__ == "__main__":
    # Instantiate the continuous STT engine session
    engine = SpeechToTextEngine(silence_limit=0.8)
    
    print_banner("ONLINE WEB SPEECH ENGINE", "Say something to start speaking... (Type 'exit application' to quit)")

    try:
        while True:
            # Capture speech transcribed inputs in a loop
            query = engine.listen_and_transcribe()
            if query:
                print_success(f"Speech Recognized: [bold highlight]{query}[/bold highlight]")
                if "exit application" in query.lower():
                    break
    except KeyboardInterrupt:
        console.print("\n[bold red]Forced Exit.[/bold red]")
    finally:
        engine.shutdown()