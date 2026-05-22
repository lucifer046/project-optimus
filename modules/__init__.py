"""
modules Package Initialization

This module configures the import boundaries for the KAYRA project's core package.
It exposes the primary components for Speech-to-Text (STT) and Text-to-Speech (TTS)
services to ensure clean integration across the assistant codebase.
"""

from .text_to_speech import TextToSpeechEngine, DynamicVoiceEngine, KokoroOnnx
from .speech_to_text import SpeechToTextEngine, recognize_speech
from .utils import get_project_root, setup_logger

# -------------------------------------------------------------------------------------------------------
#                                 Legacy Backward-Compatibility Mappings
# -------------------------------------------------------------------------------------------------------
# Exposing historical class names and methods to avoid breaking existing code blocks.
OfflineTTS = TextToSpeechEngine
LiveOfflineTTS = TextToSpeechEngine
LiveOffileTTS = TextToSpeechEngine
OnlineSpeechEngine = SpeechToTextEngine
SpeechRecognition = recognize_speech
