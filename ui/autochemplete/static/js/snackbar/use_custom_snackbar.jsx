/*
 based on https://github.com/TeamWertarbyte/material-ui-snackbar-provider/tree/efdfaed2205ebeb7f8f3f1d3b1929af9b019e2a3/stories/CustomSnackbar 
*/
import React from "react";
import { useSnackbar } from "material-ui-snackbar-provider";

export default function useCustomSnackbar() {
  const snackbar = useSnackbar();
  return React.useMemo(() => {
    const showMessage =
      (type) => (message, action, handleAction, customParameters) =>
        snackbar.showMessage(message, action, handleAction, {
          ...customParameters,
          type,
        });
    return {
      ...snackbar,
      showMessage: showMessage("info"),
      showInfo: showMessage("info"),
      showWarning: showMessage("warning"),
      showError: showMessage("error"),
      showSuccess: showMessage("success"),
    };
  }, [snackbar]);
}
