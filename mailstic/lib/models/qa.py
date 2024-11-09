from typing import TYPE_CHECKING

from sqlmodel import SQLModel, Field, Column, Relationship, Integer, String, Text

if TYPE_CHECKING:
    from lib.models.pof import POF


class QABase(SQLModel):
    question: str = Field(sa_column=Column(Text, nullable=False))
    answer: str = Field(sa_column=Column(Text, nullable=False))
    pof_slug: str = Field(foreign_key="pof.slug", nullable=False)
    name: str = Field(sa_column=Column(String(128), nullable=False))


class QA(QABase, table=True):
    id: int = Field(Integer, primary_key=True, nullable=False)

    pof: "POF" = Relationship(
        back_populates="qas", sa_relationship_kwargs={"lazy": "selectin"}
    )


class QACreate(QABase):
    pass
