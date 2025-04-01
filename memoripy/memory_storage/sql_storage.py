# mysql_storage.py
# Apache 2.0 license, Created by Awilen Bernkastel

# This is probably VERY slow.
# The only advantage of this is to avoid rewriting
# a full file for every single Interaction.

import logging

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from ..interaction import Interaction

from .sql_storage_models import MemoryOwner, Memory, Base, Embedding, Concept
from .storage import BaseStorage

logger = logging.getLogger("memoripy")

class SQLStorage(BaseStorage):

    def __init__(self, owner="Assistant", db="sqlite:///default.db", echo_sql=False):
        self.owner = None
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }
        engine = create_engine(db, echo=echo_sql)
        Base.metadata.create_all(engine)
        self.session = Session(engine)

        with self.session as s:
            owner_record = s.query(MemoryOwner).filter_by(name=owner).first()
            if owner_record:
                self.owner = owner_record
            else:
                self.owner = MemoryOwner(name=owner)
                s.add(self.owner)
                s.commit()

    def load_history(self):
        with self.session as s:
            self.owner = s.merge(self.owner)
            interactions = [x for x in self.owner.memories if x.is_long_term == False]
            self._update_history(interactions, "short_term_memory")
            interactions = [x for x in self.owner.memories if x.is_long_term == True]
            self._update_history(interactions, "long_term_memory")
        return self.history["short_term_memory"], self.history["long_term_memory"]

    def _update_history(self, interactions, memory_type):
        present_memories_uuid = {x.id for x in self.history[memory_type]}
        for interaction in interactions:
            if interaction.uuid not in present_memories_uuid:
                im = Interaction()
                im.id = interaction.uuid
                im.prompt = interaction.prompt
                im.output = interaction.output
                im.embedding = np.array([x.embedding for x in interaction.embedding]).reshape(1, -1)
                im.timestamp = interaction.timestamp
                im.concepts = [x.concept for x in interaction.concepts]
                im.access_count = interaction.access_count
                im.last_accessed = interaction.last_accessed
                im.decay_factor = interaction.decay_factor
                self.history[memory_type].append(im)

    def save_memory_to_history(self, memory_store):
        with self.session as s:
            self.owner = s.merge(self.owner)
            self._save_short_term_memory(memory_store)
            self._save_long_term_memory(memory_store)
            s.commit()

        logger.info(f"Saved interaction history to SQL db. Short-term: {len(self.history['short_term_memory'])}, Long-term: {len(self.history['long_term_memory'])}")

    def _save_short_term_memory(self, memory_store):
        interaction_ids = set(memory.id for memory in self.history["short_term_memory"])
        dict_memory = {m.uuid:m for m in self.owner.memories}
        for memory in memory_store.short_term_memory:
            if memory.id not in interaction_ids:
                new_interaction = Memory(
                    uuid=memory.id,
                    prompt=memory.prompt,
                    output=memory.output,
                    timestamp=memory.timestamp,
                    last_accessed=memory.last_accessed,
                    access_count=memory.access_count,
                    decay_factor=memory.decay_factor or 1.0,
                    is_long_term=False
                )
                for embed in memory.embedding.flatten().tolist():
                    new_interaction.embedding.append(Embedding(embedding=embed))
                for concept in list(memory.concepts):
                    new_interaction.concepts.append(Concept(concept=concept))
                self.owner.memories.append(new_interaction)

        # Remove decayed interactions
        for memory in memory_store.decayed_memory:
            if memory.id in dict_memory:
                memory_index = self.owner.memories.index([x for x in self.owner.memories if x.uuid == memory.id][0])
                self.owner.memories.pop(memory_index)

    def _save_long_term_memory(self, memory_store):
        # Any memory that will go long-term has been registered as a short-term memory to begin with.
        dict_memory = {m.uuid:m for m in self.owner.memories}
        for memory in memory_store.long_term_memory:
            dict_memory[memory.id].is_long_term = True
