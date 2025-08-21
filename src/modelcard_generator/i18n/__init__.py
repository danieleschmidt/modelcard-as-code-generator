"""Internationalization (i18n) support for model card generator."""

import json
import os
from pathlib import Path
from typing import Dict, Optional

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "ja", "zh"]

_translations: Dict[str, Dict] = {}
_current_language = DEFAULT_LANGUAGE


def load_translations() -> None:
    """Load all translation files."""
    global _translations
    
    i18n_dir = Path(__file__).parent
    
    for lang_code in SUPPORTED_LANGUAGES:
        lang_file = i18n_dir / f"{lang_code}.json"
        if lang_file.exists():
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    _translations[lang_code] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load translations for {lang_code}: {e}")


def set_language(lang_code: str) -> None:
    """Set the current language."""
    global _current_language
    
    if lang_code in SUPPORTED_LANGUAGES:
        _current_language = lang_code
    else:
        print(f"Warning: Language {lang_code} not supported, using {DEFAULT_LANGUAGE}")
        _current_language = DEFAULT_LANGUAGE


def get_language() -> str:
    """Get the current language."""
    return _current_language


def _(key: str, **kwargs) -> str:
    """Get translated text."""
    if not _translations:
        load_translations()
    
    # Navigate through nested keys (e.g., "model_card.title")
    keys = key.split(".")
    translation = _translations.get(_current_language, {})
    
    for k in keys:
        if isinstance(translation, dict) and k in translation:
            translation = translation[k]
        else:
            # Fallback to English
            fallback = _translations.get(DEFAULT_LANGUAGE, {})
            for k in keys:
                if isinstance(fallback, dict) and k in fallback:
                    fallback = fallback[k]
                else:
                    return key  # Return key if no translation found
            return str(fallback).format(**kwargs) if kwargs else str(fallback)
    
    return str(translation).format(**kwargs) if kwargs else str(translation)


def get_supported_languages() -> list:
    """Get list of supported languages."""
    return SUPPORTED_LANGUAGES.copy()


def detect_system_language() -> str:
    """Detect system language from environment."""
    # Check environment variables
    for var in ["LANG", "LANGUAGE", "LC_ALL", "LC_MESSAGES"]:
        lang = os.environ.get(var)
        if lang:
            # Extract language code (e.g., "en_US.UTF-8" -> "en")
            lang_code = lang.split("_")[0].split(".")[0].lower()
            if lang_code in SUPPORTED_LANGUAGES:
                return lang_code
    
    return DEFAULT_LANGUAGE


# Auto-detect language on import
auto_detected = detect_system_language()
set_language(auto_detected)

# Load translations on import
load_translations()


class LocalizedModelCard:
    """Model card with localization support."""
    
    def __init__(self, language: Optional[str] = None):
        if language:
            set_language(language)
        self.language = get_language()
    
    def get_section_title(self, section: str) -> str:
        """Get localized section title."""
        return _(f"model_card.{section}")
    
    def get_validation_message(self, message_key: str, **kwargs) -> str:
        """Get localized validation message."""
        return _(f"validation.{message_key}", **kwargs)
    
    def get_cli_message(self, message_key: str, **kwargs) -> str:
        """Get localized CLI message."""
        return _(f"cli.{message_key}", **kwargs)


# Global instance
localized_card = LocalizedModelCard()
