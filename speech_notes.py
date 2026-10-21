"""
Speech Notes System for VisionMate
Provides take_note, read_notes, delete_note, and clear_notes with persistence
and auto-read capability.
"""
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger(__name__)

NOTES_FILE = os.path.join(
    getattr(config, 'BASE_DIR', os.path.dirname(os.path.abspath(__file__))),
    "data",
    "notes.json"
)


class SpeechNotesManager:
    def __init__(self, notes_file: Optional[str] = None):
        self.notes_file = notes_file or NOTES_FILE
        self._ensure_dir()
        self.notes: List[Dict[str, Any]] = self._load_notes()

    def _ensure_dir(self):
        try:
            d = os.path.dirname(self.notes_file)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create dir for notes file: {e}")

    def _load_notes(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.notes_file):
            try:
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning(f"Error loading notes file: {e}")
        return []

    def _save_notes(self):
        try:
            with open(self.notes_file, 'w', encoding='utf-8') as f:
                json.dump(self.notes, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving notes file: {e}")

    def take_note(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Note cannot be empty."

        note_id = len(self.notes) + 1
        note_entry = {
            'id': note_id,
            'text': text,
            'timestamp': time.time(),
            'datetime': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.notes.append(note_entry)
        self._save_notes()
        logger.info(f"Saved note {note_id}: '{text}'")

        auto_read = getattr(config, 'AUTO_READ_NOTES', True)
        if auto_read:
            return f"Saved. You said: {text}"
        else:
            return f"Note saved: {text}"

    def read_notes(self) -> str:
        if not self.notes:
            return "You have no saved notes."

        lines = []
        for i, n in enumerate(self.notes, start=1):
            lines.append(f"Note {i}: {n.get('text', '')}")

        count = len(self.notes)
        count_str = f"You have {count} {'note' if count == 1 else 'notes'}."
        return f"{count_str} " + ". ".join(lines)

    def delete_note(self, note_identifier: Any) -> str:
        if not self.notes:
            return "You have no notes to delete."

        idx_to_remove = None

        # Check numeric index (1-based)
        if isinstance(note_identifier, int):
            target_idx = note_identifier
            if 1 <= target_idx <= len(self.notes):
                idx_to_remove = target_idx - 1
        elif isinstance(note_identifier, str):
            query = note_identifier.strip().lower()
            if query.isdigit():
                target_idx = int(query)
                if 1 <= target_idx <= len(self.notes):
                    idx_to_remove = target_idx - 1
            else:
                word_map = {
                    'one': 1, 'first': 1, 'number 1': 1, 'note 1': 1,
                    'two': 2, 'second': 2, 'number 2': 2, 'note 2': 2,
                    'three': 3, 'third': 3, 'number 3': 3, 'note 3': 3,
                    'four': 4, 'fourth': 4,
                    'five': 5, 'fifth': 5,
                }
                if query in word_map and word_map[query] <= len(self.notes):
                    idx_to_remove = word_map[query] - 1
                elif query in ['last', 'latest']:
                    idx_to_remove = len(self.notes) - 1
                else:
                    for idx, n in enumerate(self.notes):
                        if query in n.get('text', '').lower():
                            idx_to_remove = idx
                            break

        if idx_to_remove is not None:
            removed = self.notes.pop(idx_to_remove)
            # Re-index remaining notes
            for idx, n in enumerate(self.notes, start=1):
                n['id'] = idx
            self._save_notes()
            return f"Deleted note: {removed.get('text', '')}"

        return f"Could not find note matching {note_identifier}."

    def clear_notes(self) -> str:
        self.notes.clear()
        self._save_notes()
        return "All notes cleared."


# Singleton module instance and helper functions
_manager = SpeechNotesManager()


def take_note(text: str) -> str:
    """Save a voice note and return readback string."""
    return _manager.take_note(text)


def read_notes() -> str:
    """Read all saved voice notes."""
    return _manager.read_notes()


def delete_note(note_identifier: Any) -> str:
    """Delete a voice note by 1-based index or search query."""
    return _manager.delete_note(note_identifier)


def clear_notes() -> str:
    """Clear all saved voice notes."""
    return _manager.clear_notes()


def get_manager() -> SpeechNotesManager:
    """Return the global SpeechNotesManager instance."""
    return _manager


def get_notes() -> List[Dict[str, Any]]:
    """Return a copy of the list of all saved note dicts."""
    return list(_manager.notes)


def list_notes() -> List[Dict[str, Any]]:
    """Alias for get_notes."""
    return list(_manager.notes)

