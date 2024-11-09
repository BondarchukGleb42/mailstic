from typing import TYPE_CHECKING, List

from sqlmodel import SQLModel, Field, Column, String, Relationship

if TYPE_CHECKING:
    from lib.models.qa import QA


class POFBase(SQLModel):
    pass


class POF(POFBase, table=True):
    slug: str = Field(sa_column=Column(String(128), primary_key=True, nullable=False))
    qas: List["QA"] = Relationship(
        back_populates="pof", sa_relationship_kwargs={"lazy": "selectin"}
    )


class POFCreate(POFBase):
    name: str
