from autochemplete.models.treatment import TreatmentModel
from autochemplete.db import db as sa


class LabelTaskModel(sa.Model):
    __tablename__ = "label_tasks"

    id = sa.Column(sa.Integer, sa.Identity(
        start=100, cycle=True), primary_key=True)
    treatment = sa.Column(sa.Integer, sa.ForeignKey(
        TreatmentModel.id))
    image = sa.Column(sa.String)
    similar_mols_enabled = sa.Column(sa.Boolean)
    autocomplete_enabled = sa.Column(sa.Boolean)
    model_prediction = sa.Column(sa.String)
    correct_label = sa.Column(sa.String)
