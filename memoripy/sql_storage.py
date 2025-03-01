# mysql_storage.py

# This is probably VERY slow.
# The only advantage of this is to avoid rewriting
# a full file for every. Single. Interaction.

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.schema import MetaData

from .sql_storage_models import MemoryOwner
from .storage import BaseStorage

class SQLStorage(BaseStorage):

    def __init__(self, owner="Assistant", db='sqlite://'):
        self.owner = None
        self.history = {
            "short_term_memory": [],
            "long_term_memory": []
        }
        engine = create_engine(db, echo=True)
        self.session = Session(engine)

        with self.session as s:
            MetaData.create_all(engine)
            stmt = select(MemoryOwner).where(MemoryOwner.name=owner)
            for memory_owner in s.scalars(stmt):
                self.owner = memory_owner
            if self.owner is None:
                self.owner = MemoryOwner()
                self.owner.name = owner
                s.add(self.owner)
                s.commit()

    def load_history(self):
        if self.history["short_term_memory"] == []:
            with self.session as s:
                self.history["short_term_memory"] = self.owner.memories[-5:]
        # TODO... ionno.
        logging.info("No existing interaction history found in JSON. Starting fresh.")
        return [], []
            

    def save_memory_to_history(self, memory_store):
        raise NotImplementedError("The method save_memory_to_history() must be implemented.")
