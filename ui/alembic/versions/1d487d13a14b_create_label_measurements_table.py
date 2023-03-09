"""create label_measurements table

Revision ID: 1d487d13a14b
Revises: 283d0369dca9
Create Date: 2022-07-11 06:16:58.692224

"""
import uuid
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision = '1d487d13a14b'
down_revision = '283d0369dca9'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "label_measurements",
        sa.Column("id", UUID(as_uuid=True),
                  primary_key=True, default=uuid.uuid4),
        sa.Column("session", UUID(as_uuid=True),
                  sa.ForeignKey("sessions.id")),
        sa.Column("task", sa.Integer, sa.ForeignKey("label_tasks.id")),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("number_of_clicks", sa.Integer),
        sa.Column("label", sa.String),
        sa.Column("submission_path", sa.String)
    )


def downgrade():
    op.drop_table("label_measurements")
