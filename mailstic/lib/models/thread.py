import enum
from typing import TYPE_CHECKING, List, Optional

from sqlmodel import SQLModel, Field, Column, Relationship, Enum, Integer, String

if TYPE_CHECKING:
    from lib.models import Message


class RecognizeProgress(enum.Enum):
    """
    codes:
        0: не удалось узнать серийный номер за 3 сообщения, отправляем оператору;
        1: узнали серийный номер, не удалось узнать проблему за 3 сообщения;
        2: не удалось узнать серийный номер, продолжаем спрашивать;
        3: не удалось узнать тип проблемы, продолжаем спрашивать;
        4: удалось узнать все данные, отправляем оператору
    """

    UNABLE_TO_RECOGNIZE_SN = 0
    UNABLE_TO_RECOGNIZE_PROBLEM = 1
    RECOGNIZE_SN_IN_PROGRESS = 2
    RECOGNIZE_PROBLEM_IN_PROGRESS = 3
    RECOGNIZE_SUCCESSFUL = 4


class ThreadStatus(enum.Enum):
    IN_PROGRESS = 0
    SUCCESSFUL = 1


class ThreadBase(SQLModel):
    sender: str = Field(sa_column=Column(String(256), nullable=False))
    subject: str = Field(sa_column=Column(String(256), nullable=False))
    status: ThreadStatus = Field(
        sa_column=Column(
            Enum(ThreadStatus), nullable=False, default=ThreadStatus.IN_PROGRESS
        ),
        default=ThreadStatus.IN_PROGRESS,
    )
    recogize_progress: Optional[RecognizeProgress] = Field(
        sa_column=Column(Enum(RecognizeProgress)),
        default=None,
    )
    device: str = Field(sa_column=Column(String(256), nullable=True))
    serial: str = Field(sa_column=Column(String(256), nullable=True))
    problem: str = Field(sa_column=Column(String(256), nullable=True))


class Thread(ThreadBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    messages: List["Message"] = Relationship(
        back_populates="thread", sa_relationship_kwargs={"lazy": "selectin"}
    )


class ThreadCreate(ThreadBase):
    pass
