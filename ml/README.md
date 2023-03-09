# SMILES Recognition

This model supports converting images of chemical structural formulas into SMILES and other chemical representations.


## Install Enviroment

Simply run


```
conda env create --name chem_info_env --file model/utils/chem_info.yml
```

## Prediction

Put your input images into `model/utils/input_img`.

Run the following command in the path `model/`

```
python one_input_pred.py | tee log.csv
```


### Ensemble Prediction (multi-model)

Change the test path in `model/src/config.py`, and run

```
python main.py --work_type ensemble_test
```


## Results

You can find the predicted results in the folder `/model/utils/pred_img`