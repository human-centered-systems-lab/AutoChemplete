import React from "react";
import Button from "@material-ui/core/Button";

import useCustomSnackbar from "./snackbar/use_custom_snackbar";

import { submitLabelling } from "./script";

const EditorAcceptButton = () => {
  const snackbar = useCustomSnackbar();

  return (
    <Button
      variant="contained"
      onClick={() => submitLabelling({ mode: "editor", smiles: "", snackbar, submissionPath: "direct" })}
    >
      Accept Editor
    </Button>
  );
};

export default EditorAcceptButton;
