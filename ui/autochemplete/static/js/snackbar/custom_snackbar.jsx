/*
 based on https://github.com/TeamWertarbyte/material-ui-snackbar-provider/tree/efdfaed2205ebeb7f8f3f1d3b1929af9b019e2a3/stories/CustomSnackbar 
*/
import React from "react";
import { Snackbar, Button } from "@material-ui/core";
import { Alert } from "@material-ui/lab";

export default function CustomSnackbar({
  message,
  action,
  ButtonProps,
  SnackbarProps,
  customParameters,
}) {
  return (
    <Snackbar autoHideDuration={3000} {...SnackbarProps}>
      <Alert
        severity={customParameters?.type}
        action={
          action != null && (
            <Button color="inherit" size="small" {...ButtonProps}>
              {action}
            </Button>
          )
        }
      >
        {message}
      </Alert>
    </Snackbar>
  );
}
