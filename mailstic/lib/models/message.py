from typing import TYPE_CHECKING, Optional

from sqlmodel import (
    SQLModel,
    Field,
    Column,
    Relationship,
    String,
    Text,
)

if TYPE_CHECKING:
    from lib.models import Thread


class MessageBase(SQLModel):
    text: str = Field(sa_column=Column(Text, nullable=False))
    image: Optional[str] = Field(sa_column=Column(String(1024), default=None))


class Message(MessageBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    thread_id: int = Field(foreign_key="thread.id", nullable=False)
    thread: "Thread" = Relationship(
        back_populates="messages", sa_relationship_kwargs={"lazy": "selectin"}
    )


class MessageCreate(MessageBase):
    pass
