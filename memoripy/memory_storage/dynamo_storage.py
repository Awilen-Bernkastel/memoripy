# memoripy/dynamo_storage.py

import logging
import os
from dotenv import load_dotenv
from pynamodb.models import Model
from pynamodb.attributes import (
    UnicodeAttribute,
    NumberAttribute,
    ListAttribute,
    MapAttribute,
)
from pydantic import BaseModel, create_model
from memoripy import BaseStorage
from memoripy.memory_store import MemoryStore

load_dotenv()

def _get_host() -> str | None:
    return os.environ.get("MEMORIPY_DYNAMO_HOST", None)

def _get_region() -> str:
    return os.environ.get("MEMORIPY_DYNAMO_REGION", "us-east-1")

def _get_read_capacity() -> int:
    return int(os.environ.get("MEMORIPY_DYNAMO_READ_CAPACITY", "1"))

def _get_write_capacity() -> int:
    return int(os.environ.get("MEMORIPY_DYNAMO_WRITE_CAPACITY", "1"))

class ShortTermMemoryAttr(MapAttribute):
    id = UnicodeAttribute()
    prompt = UnicodeAttribute()
    output = UnicodeAttribute()
    timestamp = NumberAttribute()
    last_accessed = NumberAttribute()
    access_count = NumberAttribute()
    decay_factor = NumberAttribute()
    embedding = ListAttribute(of=NumberAttribute)
    concepts = ListAttribute(of=UnicodeAttribute)

class ShortTermMemory(BaseModel):
    id: str
    prompt: str
    output: str
    timestamp: float
    access_count: int
    last_accessed: float
    decay_factor: float
    embedding: list[float]
    concepts: list[str]

    def get(self, key, default):
        return getattr(self, key, default)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def new_from_attr(cls, attr: ShortTermMemoryAttr):
        return cls(
            id=attr.id,
            prompt=attr.prompt,
            output=attr.output,
            timestamp=attr.timestamp,
            last_accessed=attr.last_accessed,
            access_count=int(attr.access_count),
            decay_factor=attr.decay_factor,
            embedding=[float(x) for x in attr.embedding],
            concepts=[str(x) for x in attr.concepts],
        )

class LongTermMemoryAttr(MapAttribute):
    id = UnicodeAttribute()
    prompt = UnicodeAttribute()
    output = UnicodeAttribute()
    timestamp = NumberAttribute()
    last_accessed = NumberAttribute()
    access_count = NumberAttribute()
    decay_factor = NumberAttribute()
    total_score = NumberAttribute()

class LongTermMemory(BaseModel):
    id: str
    prompt: str
    output: str
    timestamp: float
    access_count: int
    last_accessed: float
    decay_factor: float
    total_score: float

    def get(self, key, default):
        return getattr(self, key, default)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def new_from_attr(cls, attr: LongTermMemoryAttr):
        return cls(
            id=attr.id,
            prompt=attr.prompt,
            output=attr.output,
            timestamp=attr.timestamp,
            last_accessed=attr.last_accessed,
            access_count=int(attr.access_count),
            decay_factor=attr.decay_factor,
            total_score=attr.total_score,
        )

class Memory(Model):
    class Meta:
        table_name = "memoripy_memory"
        region = _get_region()
        host = _get_host()

    set_id = UnicodeAttribute(hash_key=True)
    short_term_memory = ListAttribute(of=ShortTermMemoryAttr)
    long_term_memory = ListAttribute(of=LongTermMemoryAttr)

class DynamoStorage(BaseStorage):
    """Leverage DynamoDB for storage of memory interactions."""

    def __init__(self, set_id: str):
        """
        Create an instance of DynamoStorage
        Args:
            set_id: A unique identifier for the memory set - This should be
            leveraged when different memory sets make sense (e.g. when you're
            working in a multi-user system). For example, if you had a chatbot
            that handled multiple users, you would need to instantiate multiple
            instances of MemoryManager (one for each user), and you could use
            `set_id` property on DynamoStorage to indicate which user the memory
            was associated with.
        """
        if not Memory.exists():
            Memory.create_table(
                read_capacity_units=_get_read_capacity(),
                write_capacity_units=_get_write_capacity(),
                wait=True,
            )
        self.set_id = set_id

    def load_history(self):
        try:
            memory = Memory.get(self.set_id)
            short_term_memory = [self._attr_to_model(attr, ShortTermMemory) for attr in memory.short_term_memory]
            long_term_memory = [self._attr_to_model(attr, LongTermMemory) for attr in memory.long_term_memory]
            return short_term_memory, long_term_memory
        except Memory.DoesNotExist:
            return [], []

    def save_memory_to_history(self, memory_store: MemoryStore):
        history = Memory(
            set_id=self.set_id,
            short_term_memory=[],
            long_term_memory=[],
        )

        for memory in memory_store.short_term_memory:
            interaction = self._memory_to_attr(memory, ShortTermMemoryAttr)
            history.short_term_memory.append(interaction)

        for memory in memory_store.long_term_memory:
            interaction = self._memory_to_attr(memory, LongTermMemoryAttr)
            history.long_term_memory.append(interaction)

        history.save()
        logging.info(f"Saved interaction history. Short-term: {len(history.short_term_memory)}, Long-term: {len(history.long_term_memory)}")

    @staticmethod
    def _attr_to_model(attr, model_class):
        return create_model(model_class.__name__, **{k: v for k, v in attr.items()})

    def _memory_to_attr(self, memory, attr_class):
        return attr_class(
            id=memory.id,
            prompt=memory.prompt,
            output=memory.output,
            timestamp=memory.timestamp,
            last_accessed=memory.last_accessed,
            access_count=memory.access_count,
            concepts=list(memory.concepts),
            embedding=memory.embedding.flatten().tolist(),
            decay_factor=float(memory.get("decay_factor", 1.0)),
        )
