# sql_storage_models.py

# This is probably VERY slow.
# The only advantage of this is to avoid rewriting
# a full file for every. Single. Interaction.

from sqlalchemy import Integer, String, ForeignKey, Text, TIMESTAMP, List
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

class Base(DeclarativeBase):
    pass

class Memory(Base):
    __tablename__ = "memory"

    id: Mapped[Integer] = mapped_column(primary_key=True)
    owner: Mapped["MemoryOwner"] = relationship(back_populates = "memories")
    prompt: Mapped[str] = mapped_column(Text())
    output: Mapped[str] = mapped_column(Text())
    timestamp: Mapped[float] = mapped_column(TIMESTAMP())
    access_count: Mapped[int]
    decay_factor: Mapped[float]
    embeddings: Mapped[List["Embedding"]] = relationship(back_populates="memory")
    concepts: Mapped[List["Concept"]] = relationship(back_populates="memory")

class Embedding(Base):
    __tablename__ = "embedding"

    id: Mapped[int] = mapped_column(primary_key=True)
    memory_id: Mapped[int] = mapped_column(ForeignKey("memory.id"))
    embedding: Mapped[int]
    memory: Mapped["Memory"] = relationship(back_populates= "embeddings")

class Concept(Base):
    __tablename__ = "concept"

    id: Mapped[int] = mapped_column(primary_key=True)
    memory_id: Mapped[int] = mapped_column(ForeignKey("memory.id"))
    concept: Mapped[str] = mapped_column(String(30))
    memory: Mapped["Memory"] = relationship(back_populates="concepts")

class MemoryOwner(Base):
    __tablename__ = "memory_owner"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    memories: Mapped[List["Memory"]] = relationship(back_populates="owner")
