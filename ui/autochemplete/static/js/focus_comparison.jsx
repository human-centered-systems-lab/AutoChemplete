import React from "react";
import { Box, Dialog, DialogTitle, makeStyles, Typography } from "@material-ui/core";

const useStyle = makeStyles((theme) => ({
  image: {
    margin: theme.spacing(1),
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    width: "300px",
    height: "300px"
  },
}));

const FocusComparison = (props) => {
  const { onClose, open, rootImageSrc, compImageSrc } = props;

  const classes = useStyle();

  const handleClose = () => {
    onClose();
  };

  return (
    <Dialog onClose={handleClose} open={open}>
      <DialogTitle>Compare side-by-side</DialogTitle>
      <Box display="flex">
        <Box className={classes.image}>
          <img
            width="100%"
            height="100%"
            src={rootImageSrc}
            alt="Image to be labeled"
          />
          <Typography>Molecule to be labeled</Typography>
        </Box>
        <Box className={classes.image} s>
          <img
            width="100%"
            height="100%"
            src={compImageSrc}
            alt="Image in focus"
          />
          <Typography>Current Molecule</Typography>
        </Box>
      </Box>
    </Dialog>
  );
};

export default FocusComparison;
