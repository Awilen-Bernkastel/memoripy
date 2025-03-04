# mysql_storage.py
# Apache 2.0 license, Created by Awilen Bernkastel

# This is probably VERY slow.
# The only advantage of this is to avoid rewriting
# a full file for every single Interaction.

import logging

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.schema import MetaData
from .interaction_data import InteractionData

from .sql_storage_models import MemoryOwner, Memory, Base, Embedding, Concept
from .storage import BaseStorage

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

    def load_history(self, last_interactions=5):
        with self.session as s:
            self.owner = s.merge(self.owner)
            interactions = self.owner.memories[-last_interactions:] if len(self.owner.memories) >= last_interactions else self.owner.memories
            self._update_history(interactions, "short_term_memory")
            interactions = self.owner.memories[:last_interactions] if len(self.owner.memories) >= last_interactions else []
            self._update_history(interactions, "long_term_memory")
        return self.history["short_term_memory"], self.history["long_term_memory"]

    def _update_history(self, interactions, memory_type):
        present_memories_uuid = {x.id for x in self.history[memory_type]}
        for interaction in interactions:
            if interaction.uuid not in present_memories_uuid:
                im = InteractionData()
                im.id = interaction.uuid
                im.prompt = interaction.prompt
                im.output = interaction.output
                im.embedding = [x.embedding for x in interaction.embedding]
                im.timestamp = interaction.timestamp
                im.concepts = [x.concept for x in interaction.concepts]
                im.access_count = interaction.access_count
                im.decay_factor = interaction.decay_factor
                self.history[memory_type].append(im)

    def save_memory_to_history(self, memory_store):
        with self.session as s:
            self.owner = s.merge(self.owner)
            self._save_short_term_memory(memory_store, s)
            self._save_long_term_memory(memory_store, s)
            s.commit()

        logging.info(f"Saved interaction history to SQL db. Short-term: {len(self.history['short_term_memory'])}, Long-term: {len(self.history['long_term_memory'])}")

    def _save_short_term_memory(self, memory_store, session):
        interaction_ids = set(memory.id for memory in self.history["short_term_memory"])
        for memory in memory_store.short_term_memory:
            if memory.id not in interaction_ids:
                new_interaction = Memory(
                    uuid=memory.id,
                    prompt=memory.prompt,
                    output=memory.output,
                    timestamp=memory.timestamp,
                    access_count=memory.access_count,
                    decay_factor=memory.decay_factor or 1.0
                )
                for embed in memory.embedding.flatten().tolist():
                    new_interaction.embedding.append(Embedding(embedding=embed))
                for concept in list(memory.concepts):
                    new_interaction.concepts.append(Concept(concept=concept))
                self.owner.memories.append(new_interaction)
    
    def _save_long_term_memory(self, memory_store, session):
        # Save long-term memory interactions to buffer
        with session as s:
            self.owner = s.merge(self.owner)
            old_long_term_memory = [memory.uuid for memory in self.owner.long_term_memory]
            for memory in memory_store.long_term_memory:
                if memory.id not in old_long_term_memory:
                    new_interaction = Memory(
                        uuid=memory.id,
                        prompt=memory.prompt,
                        output=memory.output,
                        timestamp=memory.timestamp,
                        access_count=memory.access_count,
                        decay_factor=memory.decay_factor or 1.0
                    )
                    self.owner.long_term_memory.append(new_interaction)
            s.commit()
