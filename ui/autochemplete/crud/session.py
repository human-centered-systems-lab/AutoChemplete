import logging
from autochemplete.db import db as sa

from datetime import datetime, timedelta
from autochemplete.models.session import SessionModel
from autochemplete.models.treatment import TreatmentModel
from autochemplete.schemas.stats import Session

logger = logging.getLogger()


def create_session(treatment: str) -> Session:
    t = sa.session.query(TreatmentModel).filter(
        TreatmentModel.id == int(treatment)).first()
    if not t:
        treat = TreatmentModel(
            id=treatment, valid_until=datetime.now() + timedelta(days=365))
        try:
            sa.session.add(treat)
            sa.session.commit()
        except Exception:
            logger.warning("Rolling back treatment insert transaction.")
            sa.session.rollback()
            return {}

    s = SessionModel(start=datetime.now(), treatment=treatment)
    try:
        sa.session.add(s)
        sa.session.commit()
    except Exception:
        logger.warning("Rolling back session insert transaction.")
        sa.session.rollback()
        return {}
    return Session.from_orm(s)


def get_session(id: str) -> Session:
    s = sa.session.query(SessionModel).filter(SessionModel.id == id).first()
    return Session.from_orm(s)
