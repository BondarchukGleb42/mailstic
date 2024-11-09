from sqlmodel import SQLModel, Field, Column, Integer, Text


class QABase(SQLModel):
    question: str = Field(sa_column=Column(Text, nullable=False))
    answer: str = Field(sa_column=Column(Text, nullable=False))


class QA(QABase, table=True):
    id: int = Field(Integer, primary_key=True, nullable=False)


class QACreate(QABase):
    pass
