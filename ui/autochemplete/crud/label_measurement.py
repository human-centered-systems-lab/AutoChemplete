import logging
from autochemplete.crud.session import create_session
from autochemplete.db import db as sa
from autochemplete.models.label_measurement import LabelMeasurementModel
from autochemplete.models.label_task import LabelTaskModel
from autochemplete.models.session import SessionModel
from autochemplete.schemas.stats import LabelMeasurement

logger = logging.getLogger()


def create_label_measurement(measurement: LabelMeasurement) -> LabelMeasurement:
    l = LabelMeasurementModel(**measurement.dict(exclude={"id"}))
    s = sa.session.query(SessionModel).filter(
        SessionModel.id == measurement.session).first()
    if not s:
        s = create_session(measurement.task)
        l = LabelMeasurementModel(
            session=s.id, **measurement.dict(exclude={"session", "id"}))
    t = sa.session.query(LabelTaskModel).filter(
        LabelTaskModel.id == measurement.task).first()
    if not t:
        task = LabelTaskModel(
            id=measurement.task,
            treatment=s.treatment,
            image="",
            similar_mols_enabled=True,
            autocomplete_enabled=False,
            model_prediction="",
            correct_label=""
        )
        try:
            sa.session.add(task)
            sa.session.commit()
        except Exception as e:
            logger.warning("Rolling back task insert transaction: ", e)
            sa.session.rollback()
            return {}
    try:
        sa.session.add(l)
        sa.session.commit()
    except Exception as e:
        logger.warning("Rolling back label measurement insert transaction.")
        sa.session.rollback()
        return {}
    return LabelMeasurement.from_orm(l)
