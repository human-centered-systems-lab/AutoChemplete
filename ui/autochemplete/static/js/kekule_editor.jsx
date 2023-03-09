import React, { useEffect, useState } from "react";
import _ from "lodash";
import { Card, CardContent, CardActions, makeStyles } from "@material-ui/core";

import EditorAcceptButton from "./accept_button";
import { getCMLFromEditor, convertCMLtoSMILES } from "./script";

const useStyle = makeStyles((theme) => ({
  root: {
    minHeight: "400px",
    height: "100%",
    margin: theme.spacing(1),
    minWidth: "50%",
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between"
  },
  acceptBtn: {
    display: "flex",
    justifyContent: "flex-end",
  },
  editorContainer: {
    display: "flex",
    justifyContent: "center",
    height: "85%",
  },
}));

const KekuleEditor = (props) => {
  const classes = useStyle();
  const { initalAction, onMoleculeChange } = props;
  const [currentCML, setCurrentCML] = useState("");

  useEffect(() => {
    const id = setInterval(() => {
      setCurrentCML(getCMLFromEditor());
    }, 500);
    return () => clearInterval(id);
  }, []);

  useEffect(async () => onMoleculeChange(await convertCMLtoSMILES(currentCML)), [currentCML]);
  
  useEffect(() => {
    window.composer = new Kekule.Editor.Composer(
      document.getElementById("composer-container")
    );
    // style composer according to actual needs
    // based on 'molOnly' setup
    window.composer
      .setEnableOperHistory(true)
      .setEnableLoadNewFile(false)
      .setEnableCreateNewDoc(false)
      .setAllowCreateNewChild(false)
      .setCommonToolButtons([
        "undo",
        "redo",
        "copy",
        "cut",
        "paste",
        "zoomIn",
        "reset",
        "zoomOut",
      ]) // create all restricted common tool buttons
      .setChemToolButtons([
        "manipulate",
        "erase",
        "bond",
        "atomAndFormula",
        "ring",
        "charge",
      ]) // create only chem tool buttons related to molecule
      .setStyleToolComponentNames([
        "fontName",
        "fontSize",
        "color",
        "textDirection",
        "textAlign",
      ]); // create all default style components
    if (initalAction != null) {
      initalAction();
    }
  }, []);

  return (
    <>
      <Card className={classes.root}>
        <CardContent className={classes.editorContainer}>
          <div id="composer-container" style={{ height: "100%" }}></div>
        </CardContent>
        <CardActions className={classes.acceptBtn}>
          <EditorAcceptButton />
        </CardActions>
      </Card>
    </>
  );
};

export default KekuleEditor;
