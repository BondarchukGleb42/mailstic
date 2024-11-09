from typing import TYPE_CHECKING, Optional

from sqlmodel import (
    SQLModel,
    Field,
    Column,
    Relationship,
    Integer,
    String,
    LargeBinary,
    Text,
)

if TYPE_CHECKING:
    from lib.models import Thread


class MessageBase(SQLModel):
    text: str = Field(sa_column=Column(Text, nullable=False))
    image: Optional[bytes] = Field(
        sa_column=LargeBinary,
        default=None,
    )


class Message(MessageBase, table=True):
    id: int = Field(Integer, primary_key=True, nullable=False)

    thread_id: int = Field(foreign_key="thread.id", nullable=False)
    thread: "Thread" = Relationship(
        back_populates="messages", sa_relationship_kwargs={"lazy": "selectin"}
    )


class MessageCreate(MessageBase):
    pass
