import uuid

from sqlalchemy.dialects.postgresql import UUID

from autochemplete.models.label_task import LabelTaskModel
from autochemplete.models.session import SessionModel
from autochemplete.db import db as sa


class LabelMeasurementModel(sa.Model):
    __tablename__ = "label_measurements"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session = sa.Column(UUID(as_uuid=True), sa.ForeignKey(SessionModel.id))
    task = sa.Column(sa.ForeignKey(LabelTaskModel.id))
    duration_ms = sa.Column(sa.Integer)
    number_of_clicks = sa.Column(sa.Integer)
    label = sa.Column(sa.String)
    submission_path = sa.Column(sa.String)
