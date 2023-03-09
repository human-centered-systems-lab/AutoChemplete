from datetime import datetime
from typing import Optional
import uuid
from pydantic import BaseModel


class Session(BaseModel):
    id: Optional[uuid.UUID]
    start: datetime
    treatment: int

    class Config:
        orm_mode = True


class LabelTask(BaseModel):
    id: Optional[int]
    treatment: int
    image: str
    similar_mols_enabled: bool
    autocomplete_enabled: bool
    model_prediction: str
    correct_label: str

    class Config:
        orm_mode = True


class LabelMeasurement(BaseModel):
    id: Optional[uuid.UUID]
    session: uuid.UUID
    task: int
    duration_ms: int
    number_of_clicks: int
    label: str
    submission_path: str

    class Config:
        orm_mode = True


class InteractionEvent(BaseModel):
    id: Optional[uuid.UUID]
    time: Optional[datetime]
    type: str
    session: uuid.UUID
    before_molecule: str
    after_molecule: str
    autocompletion: Optional[str]

    class Config:
        orm_mode = True
