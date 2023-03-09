import uuid

from sqlalchemy.dialects.postgresql import UUID

from autochemplete.models.treatment import TreatmentModel
from autochemplete.db import db as sa


class SessionModel(sa.Model):
    __tablename__ = "sessions"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    start = sa.Column(sa.DateTime)
    treatment = sa.Column(sa.Integer, sa.ForeignKey(TreatmentModel.id))
