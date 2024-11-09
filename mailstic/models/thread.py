from sqlalchemy import Column, Integer

from lib.db import MariaDBBase


class Thread(MariaDBBase):
    __tablename__ = "threads"

    id = Column(Integer, primary_key=True)
