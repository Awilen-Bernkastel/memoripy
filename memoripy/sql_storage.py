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
            stmt = select(MemoryOwner).where(MemoryOwner.name==owner)
            for memory_owner in s.scalars(stmt):
                self.owner = memory_owner
            if self.owner is None:
                self.owner = MemoryOwner()
                self.owner.name = owner
                s.add(self.owner)
                s.commit()

    def load_history(self, last_interactions=5):
        with self.session as s:
            self.owner = s.merge(self.owner) # reattach owner to Session
            interactions = self.owner.memories[-last_interactions:] if len(self.owner.memories) >= last_interactions else self.owner.memories
            present_memories_uuid = [x.id for x in self.history["short_term_memory"]]
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
                    self.history["short_term_memory"].append(im)

            interactions = self.owner.memories[:last_interactions]  if len(self.owner.memories) >= last_interactions else []
            present_memories_uuid = [x.id for x in self.history["long_term_memory"]]
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
                    self.history["long_term_memory"].append(im)

        return self.history["short_term_memory"], self.history["long_term_memory"]

    def save_memory_to_history(self, memory_store):
        # Save short-term memory interactions
        with self.session as s:
            self.owner = s.merge(self.owner) # reattach owner to Session
            interaction_ids = set([x["id"] for x in self.history["short_term_memory"]])
            for idx, memory in enumerate(memory_store.short_term_memory):
                if memory["id"] not in interaction_ids:
                    interaction = Memory()
                    self.owner.memories.append(interaction)

                    interaction.uuid = memory["id"]
                    interaction.prompt = memory["prompt"]
                    interaction.output = memory["output"]

                    for embed in memory["embedding"].flatten().tolist():
                        temp = Embedding()
                        temp.embedding = embed
                        interaction.embedding.append(temp)

                    # interaction.embedding = memory["embedding"].flatten().tolist()
                    interaction.timestamp = memory["timestamp"]
                    interaction.access_count = memory["access_count"]

                    for concept in list(memory["concepts"]):
                        temp = Concept()
                        temp.concept = concept
                        interaction.concepts.append(temp)
                    # interaction.concepts = list(memory["concepts"])
                    interaction.decay_factor = memory["decay_factor"] or 1.0

            old_interaction_ids = set([x["id"] for x in self.history["long_term_memory"]])
            for idx, memory in enumerate(memory_store.long_term_memory):
                if memory["id"] in old_interaction_ids:
                    old_interaction_ids.remove(memory["id"])
            decayed_interactions = [x for x in self.owner.memories if x.id in old_interaction_ids]
            for interaction in decayed_interactions:
                s.delete(interaction)

            s.commit()

        logging.info(f"Saved interaction history to SQL db. Short-term: {len(self.history['short_term_memory'])}, Long-term: {len(self.history['long_term_memory'])}")
