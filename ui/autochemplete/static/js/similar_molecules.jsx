import React from "react";
import { makeStyles, CircularProgress, Typography, Grid, Button, Box } from "@material-ui/core";

import SimilarMoleculecard from "./similar_molecule_card";

const useStyles = makeStyles((theme) => ({
  title: {
    padding: theme.spacing(1),
  },
}));

const SimilarMolecules = (props) => {
  const classes = useStyles();
  const { molecules, loading, loadMoreSimilarMols } = props;

  return (
    <>
      <Typography variant="h6" className={classes.title}>Similar Molecules</Typography>
      <Grid
        container
        justifyContent="flex-start"
        alignItems="center"
        spacing={2}
      >
        {loading && <CircularProgress />}
        {!loading &&
          molecules &&
          molecules.map((molecule, index) => (
            <Grid item>
              <SimilarMoleculecard
                key={index}
                headerImage={`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${molecule.cid}/PNG`}
                imageAltText={"2D Molecule"}
                title={molecule.synonyms[0]}
                iupac={molecule.iupac}
                smiles={molecule.smiles}
                similarity={molecule.similarity}
                levSimilarity={molecule.levSimilarity}
              />
            </Grid>
          ))}
      </Grid>
      {!loading && <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
      >
        <Button onClick={() => loadMoreSimilarMols((prevState) => prevState + 4)}>Load more</Button>
      </Box>}
    </>
  );
};

export default SimilarMolecules;
