import React, { useState } from 'react';
import { makeStyles, } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Typography from '@material-ui/core/Typography';
import CardActionArea from '@material-ui/core/CardActionArea';
import { Button, CardActions, Tooltip } from "@material-ui/core";

import useCustomSnackbar from "./snackbar/use_custom_snackbar";

import { loadMolecule, submitLabelling, getMoleculeFromEditor } from "./script";
import FocusComparison from "./focus_comparison";

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
    minWidth: 600,
  },
  details: {
    display: 'flex',
    flexDirection: 'column',
  },
  content: {
    flex: '1 0 auto',
    "& > h5, h6": {
      width: "20rem",
    }
  },
  cover: {
    width: 222,
  },
  cardActionArea: {
    padding: 0,
    margin: 0,
    width: 222,
    height: 222,
  }
}));

const SimilarMoleculecard = (props) => {
  const classes = useStyles();
  const snackbar = useCustomSnackbar();
  const [open, setOpen] = useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };


  return (
    <>
      <Card className={classes.root}>
        <CardActionArea onClick={handleClickOpen} className={classes.cardActionArea}>
          <CardMedia
            component="img"
            image={props.headerImage}
            title={props.imageAltText}
            className={classes.cover}
          />
        </CardActionArea>
        <div className={classes.details}>
          <CardContent className={classes.content}>
            <Tooltip arrow title={props.title}>
              <Typography component="h5" variant="h5" noWrap style={props.similarity === 1.0 ? { color: "green" } : null}>
                {props.title}
              </Typography>
            </Tooltip>
            <Tooltip arrow title={props.iupac}>
              <Typography variant="subtitle1" color="textSecondary" noWrap>
                IUPAC: {props.iupac}
              </Typography>
            </Tooltip>
            <Tooltip arrow title={props.smiles}>
              <Typography variant="subtitle1" color="textSecondary" noWrap>
                SMILES: {props.smiles}
              </Typography>
            </Tooltip>
            <Typography variant="subtitle1" color="textSecondary" noWrap>
              Tanimoto Similarity: {(props.similarity * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%
            </Typography>
            <Tooltip arrow title="Levenshtein distance of SMILES strings normalized on length of longer string">
              <Typography variant="subtitle1" color="textSecondary" noWrap>
                Levenshtein Similarity: {(props.levSimilarity * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%
              </Typography>
            </Tooltip>
          </CardContent>
          <CardActions style={{ marginLeft: "auto" }}>
            <Button
              size="small"
              color="primary"
              onClick={() =>
                submitLabelling({
                  smiles: props.smiles,
                  mode: "direct",
                  snackbar,
                  submissionPath: "similar"
                })
              }
            >
              Accept
            </Button>
            <Button
              size="small"
              color="primary"
              onClick={async () => {
                fetch("/api/stats/interaction", {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                  },
                  body: JSON.stringify({
                    type: "similar_to_editor",
                    before_molecule: await getMoleculeFromEditor(),
                    after_molecule: props.smiles,
                    session: sessionId
                  }),
                }).then(async (response) => console.debug(await response.text()));

                loadMolecule(props.smiles);
              }}
            >
              Open in editor
            </Button>
          </CardActions>
        </div>
      </Card>
      <FocusComparison
        rootImageSrc={imageSrc}
        compImageSrc={props.headerImage}
        onClose={handleClose}
        open={open}
      />
    </>
  );
}

export default SimilarMoleculecard;
