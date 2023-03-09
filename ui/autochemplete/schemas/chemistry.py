from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class HashableBaseModel(BaseModel):
    class Config:
        frozen = True


class LabelRequest(BaseModel):
    resource_url: HttpUrl = Field(...)
    chemical_representation: Optional[str]
    target_url: Optional[str] = "/target"


class ChemicalFormat(str, Enum):
    inchi = "inchi"
    smiles = "smiles"
    cml = "cml"
    json = "json"
    molfile = "molfile"
    smarts = "smarts"


class SimilarityQuery(HashableBaseModel):
    molecule_a: str = Field(...)
    molecule_b: str = Field(...)


class ConversionRequest(HashableBaseModel):
    target_format: ChemicalFormat = ChemicalFormat.cml
    data: str = Field(...)


class MoleculeQuery(HashableBaseModel):
    format: ChemicalFormat = ChemicalFormat.smiles
    similar_mol_count: int = 5
    search_string: str = Field(...)


class Molecule(BaseModel):
    cid: str  # compound id from db
    iupac: Optional[str]
    smiles: str
    synonyms: List[str]


class MoleculeList(BaseModel):
    molecules: List[Molecule]


class SimilarityOutput(HashableBaseModel):
    similarity: float
