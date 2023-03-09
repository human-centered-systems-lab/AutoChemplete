import uuid
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from autochemplete.models.session import SessionModel
from autochemplete.db import db as sa


class InteractionEventModel(sa.Model):
    __tablename__ = "interaction_events"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    time = sa.Column(sa.DateTime, server_default=func.now())
    type = sa.Column(sa.String)
    session = sa.Column(UUID(as_uuid=True), sa.ForeignKey(SessionModel.id))
    before_molecule = sa.Column(sa.String)
    after_molecule = sa.Column(sa.String)
    autocompletion = sa.Column(sa.String, nullable=True)
