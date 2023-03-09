import React from "react";
import ReactDOM from "react-dom";
import levenshtein from "js-levenshtein";

import App from "./App";

if (initialMolecule === "None") {
  initialMolecule = null;
}

async function loadMolecule(smilesRepresentation) {
  const cml = await convertSMILEStoCML(smilesRepresentation);
  window.composer.setChemObj(Kekule.IO.loadFormatData(cml, "cml"));
}

// base container for kekule editor
window.composer = null;

var numberOfClicks = 0;
var start = new Date();

document.addEventListener("click", () => numberOfClicks += 1);

ReactDOM.render(<App />, document.getElementById("root"));

const submitLabelling = async ({ smiles, mode, snackbar, submissionPath }) => {
  var representation = smiles;
  if (mode === "editor") {
    try {
      representation = await getMoleculeFromEditor();
    } catch (error) {
      snackbar.showError(
        "Something went wrong. Did you submit an empty editor?"
      );
      return;
    }
  }
  
  var cont = false;
  
  if (sessionId) {
    cont = true; 
  }

  const sid = sessionId.length > 0 ? sessionId : "16fd2706-8baf-433b-82eb-8c7fada847da";
  const tid = taskId.length > 0 ? taskId :0;

  console.log(sid);

  fetch("/target", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
        label: representation, 
        number_of_clicks: numberOfClicks, 
        duration_ms: new Date() - start,
        submission_path: submissionPath,
        session: sid,
        task: tid,
    }),
  })
    .then((response) => {
      if (response.ok) {
        snackbar.showSuccess(`Exported ${representation}`);
      } else {
        snackbar.showError("Something went wrong.");
      }
    })
    .catch((error) => snackbar.showMessage(error));
    if (cont) {
      const currentUrl = new URL(window.location.href);
      var destination = currentUrl;
      if (!currentUrl.searchParams.get("task")) {
        destination = currentUrl + "?task=1";
      } else {
        const index = currentUrl.searchParams.get("task");
        destination = window.location.href.replace(/(\d+$)/, (parseInt(index) + 1).toString());
      }
      setTimeout(() => window.location.href = destination, 1000);
    }
};

const getRawMoleculeFromEditor = () => {
  if (!window.composer) {
    console.error("Cannot export from not existing composer");
    return;
  }
  let mol = window.composer.exportObjs(Kekule.Molecule)[0];
  if (!mol) {
    console.debug("Could not get molecule from editor");
    return;
  }
  return mol;
}

function getCMLFromEditor() {
  const mol = getRawMoleculeFromEditor();
  return Kekule.IO.saveFormatData(mol, "cml");
}

async function convertCMLtoSMILES(cmlMol) {
  return fetch("/api/convert", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ target_format: "smiles", data: cmlMol }),
  }).then((response) => response.text());
}

async function convertSMILEStoCML(smiles) {
  return fetch("/api/convert", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ target_format: "cml", data: smiles }),
  }).then((response) => response.text());
}

const getMoleculeFromEditor = async () => convertCMLtoSMILES(getCMLFromEditor());

const levenshteinSimilarity = (a, b) => levenshtein(a, b) / Math.max(a.length, b.length);


export { loadMolecule, submitLabelling, getMoleculeFromEditor, levenshteinSimilarity, convertCMLtoSMILES, getRawMoleculeFromEditor, convertSMILEStoCML, getCMLFromEditor };
