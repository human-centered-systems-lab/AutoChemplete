import pubchempy as pcp
from indigo import Indigo

from autochemplete.logic.utils.cache import cached

from autochemplete.schemas.chemistry import (
    ChemicalFormat,
    ConversionRequest,
    Molecule,
    MoleculeList,
    MoleculeQuery,
    SimilarityOutput,
    SimilarityQuery,
)

indigo = Indigo()


@cached
def conversion(conversion_request: ConversionRequest) -> str:
    molecule = indigo.loadMolecule(conversion_request.data)
    molecule.layout()
    molecule.standardize()
    if conversion_request.target_format == ChemicalFormat.cml:
        return molecule.cml()
    elif conversion_request.target_format == ChemicalFormat.smiles:
        return molecule.canonicalSmiles()
    elif conversion_request.target_format == ChemicalFormat.json:
        return molecule.json()
    elif conversion_request.target_format == ChemicalFormat.molfile:
        return molecule.molfile()
    elif conversion_request.target_format == ChemicalFormat.smarts:
        return molecule.json()


@cached
def similarity(query: SimilarityQuery) -> SimilarityOutput:
    mol_a = indigo.loadMolecule(query.molecule_a)
    mol_b = indigo.loadMolecule(query.molecule_b)
    return SimilarityOutput(similarity=indigo.similarity(mol_a, mol_b, "tanimoto"))


@cached
def similar_molecules(mol_query: MoleculeQuery) -> MoleculeList:
    results = pcp.get_compounds(
        identifier=mol_query.search_string,
        namespace=mol_query.format.value,
        searchtype="similarity",
        listkey_count=mol_query.similar_mol_count,
    )
    return MoleculeList(
        molecules=list(
            map(
                lambda compound: Molecule(
                    cid=compound.cid,
                    iupac=compound.iupac_name,
                    smiles=compound.isomeric_smiles,
                    synonyms=compound.synonyms,
                ),
                results,
            )
        )
    )
