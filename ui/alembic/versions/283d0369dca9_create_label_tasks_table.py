"""create label_tasks table

Revision ID: 283d0369dca9
Revises: 316631a121ec
Create Date: 2022-07-11 04:40:20.831125

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '283d0369dca9'
down_revision = '316631a121ec'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "label_tasks",
        sa.Column("id", sa.Integer, sa.Identity(
            start=100, cycle=True), primary_key=True),
        sa.Column("treatment", sa.Integer, sa.ForeignKey(
            "treatments.id")),
        sa.Column("image", sa.String),
        sa.Column("similar_mols_enabled", sa.Boolean),
        sa.Column("autocomplete_enabled", sa.Boolean),
        sa.Column("model_prediction", sa.String),
        sa.Column("correct_label", sa.String)
    )


def downgrade():
    op.drop_table("label_tasks")
