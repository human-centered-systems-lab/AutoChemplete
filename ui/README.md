# AutoChemplete UI

AutoChemplete is a Flask+React.js web application with the aim to enable interactive labeling of chemical compound images with their respective text representations.
It provides a user interface containing a chemistry editor based on Kekule.js, the image that is to be labeled and a section for results from a similarity search on the provided SMILES string.

## Structure

This application consists of a Flask-based Python web server as a base and React.js frontend in the `src/autochemplete/static/js` folder.
The React application uses npm with a custom webpack setup. The Python part uses poetry for package management. When you add new dependencies to the python project you should run `poetry export --format "requirements.txt" > requirements.txt`.

## Running the application

The application is run with docker-compose and contains a Jupyter notebook and load balancer setup next to the application. Starting up works with `AUTOCHEMPLETE_DB_PASSWORD=some_password docker compose up --build`. This makes the app available at `http://localhost:8080`.

AutoChemplete integrates into a larger application that also embeds a computer vision model for recognizing chemical formula images into SMILES strings, which can be found under smiles-cv-model.
Integration works via URL parameters on the `/label` URL.

## Parameters

- `resource_url` URL of the image being annotated
- `chemical_representation` SMILES the model recognized for the image to be annotated

In order to change the destination of label results the `/target` route in `src/autochemplete/main.py` should be modified to behave as desired.
Without further configuration labels are written to the database inside the `label_measurements` table.
