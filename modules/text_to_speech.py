# ===========================================================================================================
#                                 text_to_speech.py (Offline TTS Speech Synthesis Engine)
# ===========================================================================================================
# This module implements a premium, ultra-lightweight, 100% offline Text-to-Speech (TTS) system.
#
# Core Frameworks & Architectures:
# 1. Kokoro-82M ONNX Model: An extremely fast and premium voice model running fully optimized on the CPU,
#    saving 100% of your GPU VRAM for LLM inference.
# 2. Dynamic vocal mapping: Selects appropriate voice style vectors based on environmental variables.
# 3. Real-time Audio Streaming: Employs a background event loop thread pipeline to achieve low-latency
#    continuous physical DAC audio streaming.
# ===========================================================================================================

import os
import sys
import asyncio
import sounddevice as sd
from kokoro_onnx import Kokoro
from dotenv import dotenv_values
try:
    from .utils import print_info, print_warning, print_error, print_system, print_success, console
except ImportError:
    try:
        from modules.utils import print_info, print_warning, print_error, print_system, print_success, console
    except ImportError:
        from utils import print_info, print_warning, print_error, print_system, print_success, console


class KokoroOnnx(Kokoro):
    """
    Quantized Kokoro TTS ONNX wrapper with robust pathing and real-time streaming capability.
    Extends standard Kokoro to support a synchronous streaming generator compatible with sounddevice.
    """
    def __init__(self, model_path: str, voices_path: str):
        super().__init__(model_path, voices_path)

    def stream(self, text: str, voice: str, speed: float = 1.1, lang: str = "en-us"):
        """
        Synchronous generator wrapper for the asynchronous create_stream method.
        Yields audio chunks in real-time.
        """
        from queue import Queue
        import threading

        q = Queue()

        def run_async_loop():
            # Create a separate dedicated event loop for background voice chunk processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def consume_stream():
                    async for audio_samples, sample_rate in self.create_stream(
                        text, voice=voice, speed=speed, lang=lang, trim=True
                    ):
                        q.put((audio_samples, sample_rate))
                    q.put(None)  # Signal completion of stream

                loop.run_until_complete(consume_stream())
            except Exception as e:
                print_error(f"Error in background voice stream: {e}")
                q.put(None)
            finally:
                loop.close()

        # Execute the audio synthesis async generator on a daemon background thread
        threading.Thread(target=run_async_loop, daemon=True).start()

        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield chunk


class DynamicVoiceEngine:
    """
    Dynamic voice allocation engine configured via environment variables and supporting real-time streaming.
    """
    def __init__(self, model_filename="kokoro-v1.0.int8.onnx", voices_filename="voices-v1.0.bin"):
        # Robust path resolution
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load environment specifications
        env_vars = dotenv_values(os.path.join(project_root, ".env")) or {}
        
        # Dynamic name and fallback gender mapping
        self.assistant_name = env_vars.get("ASSISTANT_NAME", "").strip()
        if not self.assistant_name:
            self.assistant_name = "Kayra"
            gender = "female"
        else:
            gender = env_vars.get("ASSISTANT_GENDER", "Female").strip().lower()
        
        model_path = os.path.join(project_root, "models", model_filename)
        voices_path = os.path.join(project_root, "models", voices_filename)

        # Automatic fallback to standard filenames if default filenames are not present
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            alt_model_path = os.path.join(project_root, "models", "kokoro.onnx")
            alt_voices_path = os.path.join(project_root, "models", "voices.bin")
            if os.path.exists(alt_model_path) and os.path.exists(alt_voices_path):
                model_path = alt_model_path
                voices_path = alt_voices_path
            else:
                # Direct CWD check
                if os.path.exists(model_filename) and os.path.exists(voices_filename):
                    model_path = os.path.abspath(model_filename)
                    voices_path = os.path.abspath(voices_filename)

        # Verify physical model presence
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            print_error(f"Core voice matrix components missing: Check {model_path}")
            sys.exit(1)
            
        print_info("Initializing audio vectors on host CPU...")
        self.onnx = KokoroOnnx(model_path, voices_path)
        self.sample_rate = 24000
        
        # ---------------------------------------------------------
        # Dynamic Vocal Allocation Mapping
        # ---------------------------------------------------------
        if gender == "male":
            self.voice = "am_adam"  # High-quality North American Male style vector
            print_info("Vocal Cord Configuration: Male (am_adam)")
        else:
            self.voice = "af_bella"  # Ultra-realistic North American Female style vector
            print_info("Vocal Cord Configuration: Female (af_bella)")

    def speak(self, text):
        """Generates real-time audio streams directly to standard hardware drivers."""
        if not text.strip():
            return
            
        console.print(f"\n[bold magenta][{self.assistant_name} Speaking]:[/] [italic text]{text}[/]")
            
        # Strip simple markdown characters out of text before raw audio synthesis processing
        clean_text = text.replace("*", "").replace("#", "").strip()
        
        try:
            # speed=1.1 provides a clean, responsive human pacing structure
            stream_generator = self.onnx.stream(clean_text, voice=self.voice, speed=1.1)
            
            for audio_samples, sample_rate in stream_generator:
                sd.play(audio_samples, sample_rate)
                sd.wait()
        except Exception as e:
            print_error(f"Failed to stream voice output: {e}")


class TextToSpeechEngine(DynamicVoiceEngine):
    """
    Standard backward-compatible Text-to-Speech engine wrapper.
    Inherits all real-time streaming and voice mapping capabilities from DynamicVoiceEngine.
    """
    def __init__(self, model_filename="kokoro.onnx", voices_filename="voices.bin"):
        super().__init__(model_filename=model_filename, voices_filename=voices_filename)


# ===========================================================================================================
#                                  Backward Compatibility Class Aliases
# ===========================================================================================================
LiveOfflineTTS = TextToSpeechEngine
LiveOffileTTS = TextToSpeechEngine
OfflineTTS = TextToSpeechEngine


# ===========================================================================================================
#                                         Main Script Test Entrypoint
# ===========================================================================================================
if __name__ == "__main__":
    # Run a test speech synthesis sequence using the modern dynamic engine
    tts = DynamicVoiceEngine()
    tts.speak("Voice matrix active. Real-time audio streaming is fully initialized.")