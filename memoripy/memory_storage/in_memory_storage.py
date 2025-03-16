# in_memory_storage.py
# Apache 2.0 license, Modified by Awilen Bernkastel

import logging
from .storage import BaseStorage

class InMemoryStorage(BaseStorage):
    def __init__(self):
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }

    def load_history(self):
        logging.info("Loading history from in-memory storage.")
        return self.history.get("short_term_memory", []), self.history.get("long_term_memory", [])

    def save_memory_to_history(self, memory_store):
        logging.info("Saving history to in-memory storage.")
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }

        # Save short-term memory interactions
        for interaction in memory_store.short_term_memory:
            self.history["short_term_memory"].append(interaction)

        # Save long-term memory interactions
        for interaction in memory_store.long_term_memory:
            self.history["long_term_memory"].append(interaction)
