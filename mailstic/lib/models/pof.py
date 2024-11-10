from typing import TYPE_CHECKING, List, Optional

from sqlmodel import SQLModel, Field, Column, String, Relationship

if TYPE_CHECKING:
    from lib.models.qa import QA


class POFBase(SQLModel):
    name: str = Field(sa_column=Column(String(128), nullable=False))


class POF(POFBase, table=True):
    slug: Optional[str] = Field(
        sa_column=Column(String(128), primary_key=True, nullable=False),
        default=None,
    )

    qas: List["QA"] = Relationship(
        back_populates="pof",
        sa_relationship_kwargs={"lazy": "selectin"},
        cascade_delete=True,
    )


class POFCreate(POFBase):
    dataset: List[str]
