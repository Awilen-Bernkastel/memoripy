# json_storage.py

import json
import logging
import os
import pathlib

from .storage import BaseStorage

class JSONStorage(BaseStorage):
    def __init__(self, file_path="interaction_history.json"):
        self.file_path = file_path
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }

    def load_history(self):
        if len(self.history["short_term_memory"]) != 0:
            return self.history.get("short_term_memory", []), self.history.get("long_term_memory", [])
        elif os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                logging.info("Loading existing interaction history from JSON...")
                self.history = json.load(f)
                return self.history.get("short_term_memory", []), self.history.get("long_term_memory", [])
        logging.info("No existing interaction history found in JSON. Starting fresh.")
        return [], []

    def save_memory_to_history(self, memory_store):
        # Save short-term memory interactions
        interaction_ids = set([x['id'] for x in self.history['short_term_memory']])
        for idx, memory in enumerate(memory_store.short_term_memory):
            interaction = {
                'id': memory['id'],
                'prompt': memory['prompt'],
                'output': memory['output'],
                'embedding': memory_store.embeddings[idx].flatten().tolist(),
                'timestamp': memory_store.timestamps[idx],
                'access_count': memory_store.access_counts[idx],
                'concepts': list(memory_store.concepts_list[idx]),
                'decay_factor': memory.get('decay_factor', 1.0)
            }
            if interaction['id'] not in interaction_ids:
                interaction_ids += set(interaction['id'])
                self.history["short_term_memory"].append(interaction)

        # Save long-term memory interactions
        for interaction in memory_store.long_term_memory:
            if interaction['id'] not in interaction_ids:
                interaction_ids.add(interaction['id'])
                self.history["long_term_memory"].append(interaction)

        # Save the history to a file
        with open(self.file_path + "~", 'w') as f:
            json.dump(self.history, f)
        os.unlink(self.file_path)
        os.rename(self.file_path + "~", self.file_path)

        logging.info(f"Saved interaction history to JSON. Short-term: {len(self.history['short_term_memory'])}, Long-term: {len(self.history['long_term_memory'])}")
