# json_storage.py

import json
import logging
import os
import pathlib

from .interaction_data import InteractionData
from .storage import BaseStorage

class JSONStorage(BaseStorage):
    def __init__(self, file_path="interaction_history.json"):
        if file_path == "":
            file_path = "interaction_history.json"
        if not file_path.endswith(".json"):
            file_path += ".json"
        self.file_path = file_path
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }

    def load_history(self):
        if len(self.history["short_term_memory"]) != 0:
            return self.history.get("short_term_memory", []), self.history.get("long_term_memory", [])
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                logging.info("Loading existing interaction history from JSON...")
                interactions = json.load(f)
                for interaction in interactions["short_term_memory"]:
                    im = InteractionData()
                    im.id = interaction["id"]
                    im.prompt = interaction["prompt"]
                    im.output = interaction["output"]
                    im.embedding = list(interaction["embedding"])
                    im.timestamp = interaction["timestamp"]
                    im.access_count = interaction["access_count"]
                    im.concepts = interaction["concepts"]
                    im.decay_factor = interaction["decay_factor"]
                    self.history["short_term_memory"].append(im)
                for interaction in interactions.get("long_term_memory", {}):
                    im = InteractionData()
                    im.id = interaction["id"]
                    im.prompt = interaction["prompt"]
                    im.output = interaction["output"]
                    im.embedding = list(interaction["embedding"])
                    im.timestamp = interaction["timestamp"]
                    im.access_count = interaction["access_count"]
                    im.concepts = interaction["concepts"]
                    im.decay_factor = interaction["decay_factor"]
                    self.history["long_term_memory"].append(im)
            return self.history.get("short_term_memory", []), self.history.get("long_term_memory", [])
        logging.info("No existing interaction history found in JSON. Starting fresh.")
        return [], []

    def save_memory_to_history(self, memory_store):
        history = {}
        history["short_term_memory"] = []
        # Save short-term memory interactions
        interaction_ids = set([x['id'] for x in self.history['short_term_memory']])
        old_interaction_ids = set([x['id'] for x in self.history['short_term_memory']])
        for idx, memory in enumerate(memory_store.short_term_memory):
            interaction = {
                'id': memory['id'],
                'prompt': memory['prompt'],
                'output': memory['output'],
                'embedding': memory["embedding"].flatten().tolist(),
                'timestamp': memory["timestamp"],
                'access_count': memory["access_count"],
                'concepts': list(memory["concepts"]),
                'decay_factor': memory.get('decay_factor', 1.0)
            }
            history["short_term_memory"].append(interaction)
            if interaction['id'] not in interaction_ids:
                interaction_ids.add(interaction['id'])
                self.history["short_term_memory"].append(memory)
            else:
                old_interaction_ids.remove(interaction["id"])

        # Save long-term memory interactions
        interaction_ids = set([x['id'] for x in self.history['long_term_memory']])
        old_interaction_ids = set([x['id'] for x in self.history['long_term_memory']])
        for idx, memory in enumerate(memory_store.long_term_memory):
            interaction = {
                'id': memory['id'],
                'prompt': memory['prompt'],
                'output': memory['output'],
                'embedding': memory["embedding"].flatten().tolist(),
                'timestamp': memory["timestamp"],
                'access_count': memory["access_count"],
                'concepts': list(memory["concepts"]),
                'decay_factor': memory.get('decay_factor', 1.0)
            }
            history["long_term_memory"].append(interaction)
            if interaction['id'] not in interaction_ids:
                interaction_ids.add(interaction['id'])
                self.history["long_term_memory"].append(memory)
            else:
                old_interaction_ids.remove(interaction[id])

        # Save the history to a file
        with open(self.file_path + "~", 'w+') as f:
            json.dump(history, f)
        if os.path.exists(self.file_path):
            os.unlink(self.file_path)
        os.rename(self.file_path + "~", self.file_path)

        logging.info(f"Saved interaction history to JSON. Short-term: {len(self.history['short_term_memory'])}, Long-term: {len(self.history['long_term_memory'])}")
