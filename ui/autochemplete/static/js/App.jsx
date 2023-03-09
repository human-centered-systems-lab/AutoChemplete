import React from "react";
import {
  AppBar,
  Box,
  makeStyles,
  ThemeProvider,
  createTheme
} from "@material-ui/core";
import { lightBlue, yellow } from "@material-ui/core/colors";
import { SnackbarProvider } from "material-ui-snackbar-provider";
import CustomSnackbar from "./snackbar/custom_snackbar";

import SimilarMolecules from "./similar_molecules";
import ModelOutputCard from "./model_output_card";
import KekuleEditor from "./kekule_editor";
import { loadMolecule, levenshteinSimilarity } from "./script";
import { useEffect, useState } from "react";
import InputImageCard from "./image_card";

const theme = createTheme({
  palette: {
    primary: {
      main: lightBlue[600],
    },
    secondary: {
      main: yellow[300],
    },
  },
});

const useStyles = makeStyles((theme) => ({
  root: {
    display: "flex",
    flexGrow: 1,
    marginBottom: theme.spacing(2),
  },
  title: {
    flexGrow: 1,
  },
  header: {
    padding: theme.spacing(2),
  },
  fullHeight: {
    height: "100%",
  },
  linkStyle: {
    textDecoration: "none",
    color: "inherit",
    position: "relative",
    display: "inline-block",
    zIndex: 1,
  },
  beautyMargin: {
    margin: theme.spacing(2)
  },
  similarMolecules: {
    padding: `${theme.spacing(1)}px ${theme.spacing(8)}px`,
    display: "flex",
    flexDirection: "column"
  },
  mainArea: {
    display: "flex",
    justifyContent: "flex-start",
    alignItems: "center",
    paddingLeft: theme.spacing(8),
    minHeight: 400,
    height: "60vh",
  }
}));

const App = () => {
  const classes = useStyles();
  const [similarMolecules, setSimilarMolecules] = useState([]);
  const [loadingSimilar, setLoadingSimilar] = useState(false);
  const [editorMolecule, setEditorMolecule] = useState(initialMolecule);
  const [similarMolCount, setSimilarMolCount] = useState(4);

  useEffect(() => {
    if (!checkSimilarMols) {
      setLoadingSimilar(false);
      return;
    }
    setLoadingSimilar(true);
    if (!editorMolecule) {
      console.debug("Editor molecule was not available to fetch similar mols.");
      return;
    }
    fetch("/api/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        search_string: editorMolecule,
        similar_mol_count: similarMolCount,
      }),
    })
      .then((response) => response.json())
      .then(function (data) {
        setSimilarMolecules([]);
        data["molecules"].forEach(mol => {
          fetch("/api/similarity", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              molecule_a: editorMolecule,
              molecule_b: mol.smiles,
            }),
          }).then((response) => response.json())
            .then(function (data) {
              setSimilarMolecules((prevMols) => [...prevMols, { ...mol, similarity: data.similarity, levSimilarity: levenshteinSimilarity(mol.smiles, editorMolecule) }]);
              setSimilarMolecules((prevMols) => prevMols.sort((a, b) => a.similarity > b.similarity ? -1 : a.similarity < b.similarity ? 1 : 0));
            })
        });
        setLoadingSimilar(false);
      });
  }, [editorMolecule, similarMolCount]);

  return (
    <ThemeProvider theme={theme}>
      <SnackbarProvider
        SnackbarComponent={CustomSnackbar}
        SnackbarProps={{
          anchorOrigin: { vertical: "bottom", horizontal: "left" },
        }}
      >
        <Box className={classes.root}>
          <AppBar position="static" className={classes.header}>

            <Box display="flex">
              <Box minWidth="64px" />
              <a
                className={classes.linkStyle}
                href={window.location.origin}
              >
                <span style={{ display: "inline-block" }}>
                  <img
                    style={{ position: "relative", zIndex: -1, fill: "white" }}
                    src="/static/assets/autochemplete-logo.svg"
                    type="image/svg+xml"
                    width="150px">
                  </img>
                </span>
              </a>
            </Box>
          </AppBar>
        </Box>
        <Box className={classes.mainArea}>
          <Box display="flex" flexDirection="column" height="100%">
            <InputImageCard
              headerImage={imageSrc}
              imageAltText={"Molecule image to be labeled"}
            />
            <Box minHeight="0.5vh" />
            {initialMolecule !== null && initialMolecule !== "" ? <ModelOutputCard title={initialMolecule} /> : null}
          </Box>
          <Box minWidth={"5%"} />
          <KekuleEditor
            initalAction={() => loadMolecule(initialMolecule)}
            onMoleculeChange={setEditorMolecule} />
        </Box>
        {checkSimilarMols ?
          <Box className={classes.similarMolecules}>
            <SimilarMolecules molecules={similarMolecules} loading={loadingSimilar} loadMoreSimilarMols={setSimilarMolCount} />
          </Box> : null
        }

      </SnackbarProvider>
    </ThemeProvider>
  );
};

export default App;
