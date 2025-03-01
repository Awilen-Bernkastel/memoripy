# mysql_storage.py

# This is probably VERY slow.
# The only advantage of this is to avoid rewriting
# a full file for every. Single. Interaction.

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.schema import MetaData

from .sql_storage_models import MemoryOwner, Memory
from .storage import BaseStorage

class SQLStorage(BaseStorage):

    def __init__(self, owner="Assistant", db="sqlite://", echo_sql=False):
        self.owner = None
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }
        self.session = Session(create_engine(db, echo=echo_sql))

        with self.session as s:
            MetaData.create_all(self.session)
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
            self.history["short_term_memory"] = self.owner.memories[-last_interactions:] if len(self.owner.memories) >= last_interactions else self.owner.memories
            self.history["long_term_memory"]  = self.owner.memories[:last_interactions]  if len(self.owner.memories) >= last_interactions else []
        return self.history["short_term_memory"], self.history["long_term_memory"]

    def save_memory_to_history(self, memory_store):
        # Save short-term memory interactions
        with self.session as s:
            interaction_ids = set([x["id"] for x in self.history["short_term_memory"]])
            for idx, (memory, embedding, timestamp, access_count, concepts) in enumerate(zip(memory_store.short_term_memory, memory_store.embeddings, memory_store.access.counts, memory_store.concepts_list)):
                if memory["id"] not in interaction_ids:
                    interaction = Memory()
                    self.owner.memories.append(interaction)

                    interaction.id = memory["id"]
                    interaction.prompt = memory["prompt"]
                    interaction.output = memory["output"]
                    interaction.embedding = embedding.flatten().tolist()
                    interaction.timestamp = timestamp
                    interaction.access_count = access_count
                    interaction.concepts = list(concepts)
                    interaction.decay_factor = memory.get("decay_factor", 1.0)

            old_interaction_ids = set([x["id"] for x in self.history["long_term_memory"]])
            for idx, memory in enumerate(memory_store.long_term_memory):
                if memory["id"] in old_interaction_ids:
                    old_interaction_ids.remove(memory["id"])
            decayed_interactions = [x for x in self.owner.memories if x.id in old_interaction_ids]
            for interaction in decayed_interactions:
                s.delete(interaction)

            s.commit()

        logging.info(f"Saved interaction history to SQL db. Short-term: {len(self.history['short_term_memory'])}, Long-term: {len(self.history['long_term_memory'])}")
