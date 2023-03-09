"""create treatment table

Revision ID: a68a008f4c6c
Revises: 
Create Date: 2022-07-11 03:55:34.715423

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a68a008f4c6c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "treatments",
        sa.Column("id", sa.Integer, sa.Identity(
            start=100, cycle=True), primary_key=True),
        sa.Column("valid_until", sa.DateTime)
    )


def downgrade():
    op.drop_table("treatments")
