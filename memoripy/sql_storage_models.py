# sql_storage_models.py
# Apache 2.0 license, Created by Awilen Bernkastel

# This is probably VERY slow.
# The only advantage of this is to avoid rewriting
# a full file for every single Interaction.

from sqlalchemy import Integer, String, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from typing import List

class Base(DeclarativeBase):
    pass

class Memory(Base):
    __tablename__ = "memory"

    id: Mapped[int] = mapped_column(primary_key=True)
    uuid: Mapped[str] = mapped_column(unique=True)
    prompt: Mapped[str] = mapped_column(Text())
    output: Mapped[str] = mapped_column(Text())
    timestamp: Mapped[float] = mapped_column(Float())
    access_count: Mapped[int]
    decay_factor: Mapped[float]
    owner_id: Mapped[int] = mapped_column(ForeignKey("memory_owner.id"))
    embedding: Mapped[List["Embedding"]] = relationship(back_populates="memory")
    concepts: Mapped[List["Concept"]] = relationship(back_populates="memory")
    owner: Mapped["MemoryOwner"] = relationship(back_populates = "memories")
    is_long_term: Mapped[bool] = mapped_column(Boolean())

class Embedding(Base):
    __tablename__ = "embedding"

    id: Mapped[int] = mapped_column(primary_key=True)
    memory_id: Mapped[int] = mapped_column(ForeignKey("memory.id"))
    embedding: Mapped[float]

    memory: Mapped["Memory"] = relationship(back_populates= "embedding")

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
