from typing import TYPE_CHECKING, Optional

from sqlmodel import (
    SQLModel,
    Field,
    Column,
    Relationship,
    Integer,
    String,
    Text,
    ForeignKey,
)

if TYPE_CHECKING:
    from lib.models.pof import POF


class QABase(SQLModel):
    question: str = Field(sa_column=Column(Text, nullable=False))
    answer: str = Field(sa_column=Column(Text, nullable=False))


class QA(QABase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pof_slug: str = Field(sa_column=Column(ForeignKey("pof.slug"), nullable=False))

    pof: "POF" = Relationship(
        back_populates="qas", sa_relationship_kwargs={"lazy": "selectin"}
    )


class QACreate(QABase):
    pass
