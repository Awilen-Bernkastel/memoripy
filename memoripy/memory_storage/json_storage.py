# json_storage.py
import json
import logging
import os

from ..interaction import Interaction
from .storage import BaseStorage

logger = logging.getLogger("memoripy")

class JSONStorage(BaseStorage):
    def __init__(self, file_path="interaction_history.json"):
        self.file_path = file_path if file_path.endswith(".json") else f"{file_path}.json"
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }

    def load_history(self):
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                logger.info("Loading existing interaction history from JSON...")
                history = json.load(f)
                for im in history['short_term_memory']:
                    self.history['short_term_memory'].append(self._deserialize_interaction(im))
                for im in history['long_term_memory']:
                    self.history['long_term_memory'].append(self._deserialize_interaction(im))
        else:
            logger.info("No existing interaction history found in JSON. Starting fresh.")
        return self.history.get("short_term_memory", []), self.history.get("long_term_memory", [])

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

        logger.info(f"Saved interaction history to JSON. Short-term: {len(memory_store.short_term_memory)}, Long-term: {len(memory_store.long_term_memory)}")

    def _serialize_interaction(self, memory):
        return {
            'id': memory.id,
            'prompt': memory.prompt,
            'output': memory.output,
            'embedding': memory.embedding.flatten().tolist(),
            'timestamp': memory.timestamp,
            'last_accessed': memory.last_accessed,
            'access_count': memory.access_count,
            'concepts': list(memory.concepts),
            'decay_factor': memory.get('decay_factor', 1.0)
        }

    def _deserialize_interaction(self, memory):
        return Interaction(
            id=memory['id'],
            prompt=memory['prompt'],
            output=memory['output'],
            embedding=memory['embedding'],
            timestamp=memory['timestamp'],
            last_accessed=memory['last_accessed'],
            access_count=memory['access_count'],
            concepts=list(memory['concepts']),
            decay_factor=memory.get('decay_factor', 1.0)
        )