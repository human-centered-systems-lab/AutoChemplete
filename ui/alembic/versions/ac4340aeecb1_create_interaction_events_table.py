"""create interaction_events table

Revision ID: ac4340aeecb1
Revises: 1d487d13a14b
Create Date: 2022-07-11 06:45:43.579762

"""
import uuid
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision = 'ac4340aeecb1'
down_revision = '1d487d13a14b'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "interaction_events",
        sa.Column("id", UUID(as_uuid=True),
                  primary_key=True, default=uuid.uuid4),
        sa.Column("time", sa.DateTime, server_default=func.now()),
        sa.Column("type", sa.String),
        sa.Column("session", UUID(as_uuid=True),
                  sa.ForeignKey("sessions.id")),
        sa.Column("before_molecule", sa.String),
        sa.Column("after_molecule", sa.String),
        sa.Column("autocompletion", sa.String, nullable=True)
    )


def downgrade():
    op.drop_table("interaction_events")
