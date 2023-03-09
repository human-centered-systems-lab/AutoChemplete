import logging

from autochemplete.models.interaction_event import InteractionEventModel
from autochemplete.db import db as sa
from autochemplete.schemas.stats import InteractionEvent

logger = logging.getLogger()


def create_interaction_event(event: InteractionEvent) -> InteractionEvent:
    e = InteractionEventModel(**event.dict())
    try:
        sa.session.add(e)
        sa.session.commit()
    except Exception as e:
        logger.warning("Rolling back interaction event insert transaction.")
        sa.session.rollback()
    return InteractionEvent.from_orm(e)
