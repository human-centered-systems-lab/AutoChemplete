"""create sessions table

Revision ID: 316631a121ec
Revises: a68a008f4c6c
Create Date: 2022-07-11 04:03:37.531631

"""
import uuid
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision = '316631a121ec'
down_revision = 'a68a008f4c6c'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  default=uuid.uuid4),
        sa.Column("start", sa.DateTime),
        sa.Column("treatment", sa.Integer,
                  sa.ForeignKey("treatments.id"))
    )


def downgrade():
    op.drop_table("sessions")
