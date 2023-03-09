from typing import List
from autochemplete.models.label_task import LabelTaskModel
from autochemplete.schemas.stats import LabelTask
from autochemplete.db import db as sa


def get_label_tasks_for_treatment(treatment: int) -> List[LabelTask]:
    return [LabelTask.from_orm(task)
            for task in sa.session
            .query(LabelTaskModel)
            .filter(LabelTaskModel.treatment == treatment)
            .all()]
