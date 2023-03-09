from autochemplete.db import db as sa


class TreatmentModel(sa.Model):
    __tablename__ = "treatments"

    id = sa.Column(sa.Integer, sa.Identity(
        start=100, cycle=True), primary_key=True)
    valid_until = sa.Column(sa.DateTime)
