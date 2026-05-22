# ┌────────────────────────────────────────────────────────────────────────┐
# │                         automation_windows.py                          │
# │                  System Control & Keystroke Core                       │
# └────────────────────────────────────────────────────────────────────────┘
"""
This module acts as the "Hands" of the AI, executing high-priority system automation scripts.

Core Frameworks & Architectures:
1. Hardware-Level Input Injection: Simulates low-level keyboard inputs globally across any application.
2. Asynchronous Task Concurrency: Leverages asyncio thread pooling to run multiple automations in parallel.
3. Win32 Windows Subsystem Tracing: Safely targets, focuses, and destroys specific tabs or standalone windows.
4. Centralized Supply Linkage: Drops local redundant clients, routing text generation tasks back to the engine.
"""

import os
import sys
import time
import ctypes
import platform
import tempfile
import asyncio
import subprocess
import requests
import webbrowser
import keyboard
import pyautogui
from pynput.keyboard import Controller

# Suppress noisy startup output of third-party loaders (e.g. AppOpener, pywhatkit)
try:
    _old_stdout = sys.stdout
    _old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    from AppOpener import close, open as appopen
    from pywhatkit import search, playonyt
finally:
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = _old_stdout
    sys.stderr = _old_stderr

from bs4 import BeautifulSoup
from dotenv import dotenv_values

# Robust relative path import vectors for unified ecosystem execution
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

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if __file__ else "."
env_vars = dotenv_values(os.path.join(root, ".env")) or {}
username = env_vars.get("USERNAME", "User")

# Share the centralized engine instance for content generation pipelines
engine = CentralizedLLMEngine()
virtual_keyboard = Controller()

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
messages_cache = []

# ┌────────────────────────────────────────────────────────────────────────┐
# │                GLOBAL DESKTOP TYPE HARDWARE INJECTION                  │
# └────────────────────────────────────────────────────────────────────────┘

def global_desktop_type(spoken_text):
    """
    Simulates hardware keyboard keystrokes instantly at the current active mouse cursor.
    Uses clipboard fallback logic to ensure complex strings or Hinglish phonetics transfer safely.
    """
    cleaned_text = spoken_text.strip()
    if cleaned_text.lower().startswith("write "):
        payload = spoken_text[6:]
    elif cleaned_text.lower().startswith("type "):
        payload = spoken_text[5:]
    else:
        payload = spoken_text

    print_info(f"Injecting simulated desktop keystrokes: '{payload[:30]}...'")
    
    # 0.5s safety window to allow focus stabilization
    time.sleep(0.5)
    try:
        # Utilizing fast write sequence intervals
        pyautogui.write(payload, interval=0.005)
        return True
    except Exception as e:
        print_error(f"Hardware-level text injection failure: {e}")
        return False

# ┌────────────────────────────────────────────────────────────────────────┐
# │                       AUTOMATION CORE ENGINE                           │
# └────────────────────────────────────────────────────────────────────────┘

def WebSearch(query):
    """Executes default search engine tracking."""
    search(query)
    return True

def Content(topic):
    """
    Generates programming logic or scripts via the Centralized Engine, 
    persists it into a system temp cache, and targets notepad to display it.
    """
    global messages_cache
    topic_clean = topic.replace("content", "").strip()
    print_info(f"Synthesizing dedicated script asset for: '{topic_clean[:20]}...'")

    system_prompt = f"You are a professional content writer and expert software programmer. Write high-quality code, emails, or text for {username}. Do not include markdown wraps or conversational fluff."
    messages_cache.append({"role": "user", "content": topic_clean})

    api_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": topic_clean}
    ]

    # Leverage unified streaming pipeline directly
    generated_buffer = ""
    for chunk in engine.generate_chat_stream(api_payload):
        generated_buffer += chunk

    messages_cache.append({"role": "assistant", "content": generated_buffer})

    # Save to disk securely
    temp_dir = tempfile.gettempdir()
    file_name = f"MYSTERY_Output_{int(time.time())}.txt"
    filepath = os.path.join(temp_dir, file_name)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(generated_buffer)

    print_success(f"Content generation committed to temporary cache file.")
    subprocess.Popen(["notepad.exe", filepath])
    return True

def YoutubeSearch(topic):
    """Opens target query matrices within YouTube index parameters."""
    webbrowser.open(f"https://www.youtube.com/results?search_query={topic}")
    return True

def PlayYoutube(query):
    """Dispatches instant video streaming links via pywhatkit."""
    playonyt(query)
    return True

def OpenApp(app):
    """Opens local window executables or falls back to scraping links instantly."""
    app_target = app.lower().strip()
    print_info(f"Targeting system execution paths for: '{app_target}'")

    if app_target in ["file explorer", "file manager", "my computer", "this pc", "explorer"]:
        os.startfile("explorer")
        return True

    try:
        appopen(app_target, match_closest=True, output=False, throw_error=True)
        return True
    except:
        print_warning(f"Local app shortcut not resolved. Running fast web scraping link extraction...")
        try:
            search_url = f"https://www.google.com/search?q={app_target}"
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(search_url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                link_node = soup.find("a", {"jsname": "UWckNb"})
                resolved_url = link_node.get("href") if link_node else f"https://www.google.com/search?q={app_target}"
                
                # Enforce native Chrome execution pathways if present on Windows
                chrome_paths = [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
                ]
                for path in chrome_paths:
                    if os.path.exists(path):
                        subprocess.run([path, resolved_url])
                        return True
                webbrowser.open(resolved_url)
                return True
        except Exception as web_err:
            print_error(f"Web fallback routing layer failed: {web_err}")
            return False

def CloseApp(app):
    """
    Advanced target window trace mechanism.
    Intercepts window title attributes via low-level Win32 tracking callbacks.
    """
    app_target = app.lower().strip()
    
    if "file explorer" in app_target or app_target == "explorer":
        cmd = "powershell -Command \"(New-Object -ComObject Shell.Application).Windows() | ForEach-Object { $_.Quit() }\""
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
        return True

    if "settings" in app_target:
        subprocess.run("taskkill /f /im SystemSettings.exe", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True

    try:
        from ctypes import windll, create_unicode_buffer
        WM_CLOSE = 0x0010
        target_hwnds = []

        def enum_windows_proc(hwnd, _):
            if windll.user32.IsWindowVisible(hwnd):
                length = windll.user32.GetWindowTextLengthW(hwnd)
                buf = create_unicode_buffer(length + 1)
                windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                if app_target in buf.value.lower():
                    target_hwnds.append((hwnd, buf.value))
            return True

        enum_cb = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)(enum_windows_proc)
        windll.user32.EnumWindows(enum_cb, 0)

        if target_hwnds:
            for hwnd, title in target_hwnds:
                is_browser = any(b in title.lower() for b in ["chrome", "edge", "firefox", "brave", "opera"])
                # Safe close tabs if browser window context maps directly
                if is_browser and not any(ext in app_target for ext in ["chrome", "edge", "browser"]):
                    print_warning(f"Isolating active browser window tab focus hook: '{title[:20]}'")
                    windll.user32.ShowWindow(hwnd, 9) # Restore window state
                    windll.user32.SetForegroundWindow(hwnd)
                    time.sleep(0.3)
                    keyboard.press_and_release("ctrl+w")
                else:
                    print_info(f"Sending hardware-level close interrupt signals to: '{title[:20]}'")
                    windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            return True
    except Exception as e:
        print_error(f"Advanced Win32 title callback tracing crashed: {e}")

    # Aggressive fallback logic matrix for non-critical targets
    critical_blocks = ["explorer", "winlogon", "services", "svchost", "python", "py"]
    if app_target in critical_blocks:
        return False
    
    if app_target.endswith(".exe"):
        subprocess.run(f"taskkill /f /im {app_target}", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    else:
        subprocess.run(f"taskkill /f /im {app_target}.exe", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return True

def _set_brightness(target):
    """Directly sets monitor brightness to a specific percentage level."""
    try:
        target = max(0, min(100, target))
        set_cmd = f'powershell -Command "(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {target})"'
        subprocess.run(set_cmd, shell=True)
        print_success(f"System monitor panel luminance set directly to {target}%")
    except Exception as e:
        print_error(f"Direct monitor brightness adjustment failed: {e}")

def ExecuteCommand(command):
    """Hardware command registry processing matrix."""
    cmd = command.lower().strip()
    
    if "mute" in cmd:
        keyboard.press_and_release("volume mute")
    elif "volume up" in cmd or "increase volume" in cmd or "volume increase" in cmd or "raise volume" in cmd:
        for _ in range(5): keyboard.press_and_release("volume up")
    elif "volume down" in cmd or "decrease volume" in cmd or "volume decrease" in cmd or "lower volume" in cmd:
        for _ in range(5): keyboard.press_and_release("volume down")
    elif "lock" in cmd:
        ctypes.windll.user32.LockWorkStation()
    elif "sleep" in cmd or "turn off screen" in cmd:
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    elif "shutdown" in cmd:
        os.system("shutdown /s /t 0")
    elif "restart" in cmd:
        os.system("shutdown /r /t 0")
    elif "brightness" in cmd:
        import re
        match = re.search(r'(\d+)', cmd)
        if match:
            target_val = int(match.group(1))
            _set_brightness(target_val)
        elif "up" in cmd or "increase" in cmd or "raise" in cmd:
            _adjust_brightness(15)
        elif "down" in cmd or "decrease" in cmd or "lower" in cmd:
            _adjust_brightness(-15)
        else:
            print_warning(f"Failed to match brightness operation parameters: {cmd}")
    else:
        print_warning(f"System execution command route failed to match target pattern: {cmd}")
    return True

def _adjust_brightness(delta):
    """Executes high-privilege WMI monitor configuration arrays."""
    try:
        get_cmd = 'powershell -Command "(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightness).CurrentBrightness"'
        res = subprocess.run(get_cmd, capture_output=True, text=True, shell=True)
        current = int(res.stdout.strip()) if res.stdout.strip() else 50
        target = max(0, min(100, current + delta))
        _set_brightness(target)
    except Exception as e:
        print_error(f"Monitor instrumentation brightness adjustments failed: {e}")

# ┌────────────────────────────────────────────────────────────────────────┐
# │                 ASYNCHRONOUS ORCHESTRATOR CHANNELS                     │
# └────────────────────────────────────────────────────────────────────────┘

async def translate_and_execute(commands: list[str]):
    """Iterates through dynamic array strings scheduling concurrent automation execution targets."""
    tasks = []

    for command in commands:
        cmd_str = command.strip()
        cmd_lower = cmd_str.lower()
        
        # 1. Catch Global Type Injections First
        if cmd_lower.startswith("write ") or cmd_lower.startswith("type "):
            tasks.append(asyncio.to_thread(global_desktop_type, cmd_str))

        # 2. Match standard processing automation headers
        elif cmd_lower.startswith("open "):
            tasks.append(asyncio.to_thread(OpenApp, cmd_str.removeprefix("open ")))
        elif cmd_lower.startswith("close "):
            tasks.append(asyncio.to_thread(CloseApp, cmd_str.removeprefix("close ")))
        elif cmd_lower.startswith("play "):
            tasks.append(asyncio.to_thread(PlayYoutube, cmd_str.removeprefix("play ")))
        elif cmd_lower.startswith("content "):
            tasks.append(asyncio.to_thread(Content, cmd_str.removeprefix("content ")))
        elif cmd_lower.startswith("youtube search "):
            tasks.append(asyncio.to_thread(YoutubeSearch, cmd_str.removeprefix("youtube search ")))
        elif cmd_lower.startswith("google search "):
            tasks.append(asyncio.to_thread(WebSearch, cmd_str.removeprefix("google search ")))
        elif cmd_lower.startswith("system "):
            tasks.append(asyncio.to_thread(ExecuteCommand, cmd_str.removeprefix("system ")))
            
        # 3. Direct matching pass-through handling
        elif any(k in cmd_lower for k in ["volume", "brightness", "mute", "lock", "shutdown", "restart", "sleep"]):
            tasks.append(asyncio.to_thread(ExecuteCommand, cmd_str))
        else:
            print_warning(f"Skipping unmapped system loop command token: '{cmd_str}'")

    if tasks:
        # Launch concurrent worker channels instantly across system layers
        await asyncio.gather(*tasks)

async def Automation(commands: list[str]):
    """Public wrapper to access asynchronous macro-processing tasks."""
    await translate_and_execute(commands)
    return True

# ┌────────────────────────────────────────────────────────────────────────┐
# │                         DIAGNOSTIC TEST NODE                           │
# └────────────────────────────────────────────────────────────────────────┘

if __name__ == "__main__":
    print_banner("KAYRA WINDOWS MACRO CORE", "High-Privilege System Automation Node")
    print_success("Asynchronous execution pools initialized successfully. Automation layer active.")

    async def test_loop():
        while True:
            try:
                cmd_input = await asyncio.to_thread(input, "\nAutomation Command > ")
                if not cmd_input.strip(): continue
                if cmd_input.lower() in ["exit", "quit", "bye"]: break
                
                await Automation([cmd_input])
            except (KeyboardInterrupt, asyncio.CancelledError):
                print_system("Manual interrupt. Exiting diagnostic processor.")
                break
            except Exception as e:
                print_error(f"Diagnostic processor halted: {e}")

    try:
        asyncio.run(test_loop())
    except KeyboardInterrupt:
        pass