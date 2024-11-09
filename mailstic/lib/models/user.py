from sqlmodel import SQLModel, Field, Column, String


class User(SQLModel, table=True):
    login: str = Field(sa_column=Column(String(128), primary_key=True, nullable=False))
    password: str = Field(sa_column=Column(String(128), nullable=False))
