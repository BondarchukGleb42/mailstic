"""improves

Revision ID: 33601f627e2b
Revises: d513b8aaad43
Create Date: 2024-11-10 09:28:32.537880

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = '33601f627e2b'
down_revision: Union[str, None] = 'd513b8aaad43'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('message', schema=None) as batch_op:
        batch_op.alter_column('image',
               existing_type=sa.BLOB(),
               type_=sa.String(length=1024),
               existing_nullable=True)

    with op.batch_alter_table('thread', schema=None) as batch_op:
        batch_op.add_column(sa.Column('sender', sa.String(length=256), nullable=False))

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('thread', schema=None) as batch_op:
        batch_op.drop_column('sender')

    with op.batch_alter_table('message', schema=None) as batch_op:
        batch_op.alter_column('image',
               existing_type=sa.String(length=1024),
               type_=sa.BLOB(),
               existing_nullable=True)

    # ### end Alembic commands ###
