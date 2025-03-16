# json_storage.py
import json
import logging
import os

from ..interaction_data import InteractionData
from .storage import BaseStorage

class JSONStorage(BaseStorage):
    def __init__(self, file_path="interaction_history.json"):
        self.file_path = file_path if file_path.endswith(".json") else f"{file_path}.json"
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                logging.info("Loading existing interaction history from JSON...")
                interactions = json.load(f)
                self.history.update(interactions)
        else:
            logging.info("No existing interaction history found in JSON. Starting fresh.")
        return self.history.get("short_term_memory", []), self.history.get("long_term_memory", [])

    def _load_interactions(self, interactions, memory_type):
        for interaction in interactions:
            im = InteractionData()
            for key, value in interaction.items():
                setattr(im, key, value)
            self.history[memory_type].append(im)

    def save_memory_to_history(self, memory_store):
        history = {
            "short_term_memory": [self._serialize_interaction(m) for m in memory_store.short_term_memory],
            "long_term_memory": [self._serialize_interaction(m) for m in memory_store.long_term_memory]
        }
        with open(self.file_path + "~", 'w+') as f:
            json.dump(history, f)

        if os.path.exists(self.file_path):
            os.unlink(self.file_path)
        os.rename(self.file_path + "~", self.file_path)

        logging.info(f"Saved interaction history to JSON. Short-term: {len(memory_store.short_term_memory)}, Long-term: {len(memory_store.long_term_memory)}")

    def _serialize_interaction(self, memory):
        return {
            'id': memory['id'],
            'prompt': memory['prompt'],
            'output': memory['output'],
            'embedding': memory["embedding"].flatten().tolist(),
            'timestamp': memory["timestamp"],
            'access_count': memory["access_count"],
            'concepts': list(memory["concepts"]),
            'decay_factor': memory.get('decay_factor', 1.0)
        }
