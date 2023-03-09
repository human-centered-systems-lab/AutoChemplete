import logging
from flask import Flask, render_template, request, session
from autochemplete.crud.interaction_event import create_interaction_event
from autochemplete.crud.label_measurement import create_label_measurement

from autochemplete.db import db
from autochemplete.crud.session import create_session, get_session
from autochemplete.logic.utils.cache import get_cache, cache_stats, clear_cache
from autochemplete.logic.chemistry import conversion, similar_molecules, similarity
from autochemplete.schemas.chemistry import ConversionRequest, LabelRequest, MoleculeQuery, SimilarityQuery
from autochemplete.crud.label_task import get_label_tasks_for_treatment
from autochemplete.schemas.stats import InteractionEvent, LabelMeasurement

app = Flask(__name__)
db.init_app(app)
app.config.from_object("autochemplete.config.BaseConfig")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "super secret autochemplete string"


@app.route("/healthz")
def health_check():
    return {"base_server": True, "cache": get_cache().ping()}


@app.route("/")
def base_route():
    return render_template("home.html")


@app.route("/label", methods=["GET"])
def labelling():
    if request.args:
        request_model = LabelRequest(**request.args)
        app.logger.info(request_model)
        return render_template(
            "label.html",
            model=request_model,
            checkSimilarMols=True,
            checkAutocompletions=True,
        )
    else:
        return render_template("label-form.html")


@app.route("/eval/<treatment_id>", methods=["GET"])
def evaluation(treatment_id):
    if not "evaluation" in session:
        new_session = create_session(treatment_id)
        session["evaluation"] = new_session.id

    args = request.args
    screen = args.get("task", default=0, type=int)
    s = get_session(session["evaluation"])
    tasks = get_label_tasks_for_treatment(treatment_id)
    if screen > len(tasks) - 1:
        return "Done 🥳 Thank you!"
    task = tasks[screen]
    return render_template(
        "label.html",
        model=LabelRequest(resource_url=task.image,
                           chemical_representation=task.model_prediction),
        checkSimilarMols=task.similar_mols_enabled, checkAutocompletions=task.autocomplete_enabled,
        sessionId=s.id,
        taskId=task.id
    )


@app.route("/api/convert", methods=["POST"])
def convert():
    if request.json:
        request_model = ConversionRequest(**request.json)
        app.logger.info(request_model)
        return conversion(request_model)


@app.route("/api/similarity", methods=["POST"])
def similarity_mol():
    if request.json:
        request_model = SimilarityQuery(**request.json)
        app.logger.info(request_model)
        return similarity(request_model).json()


@app.route("/api/search", methods=["POST"])
def search():
    if request.json:
        request_model = MoleculeQuery(**request.json)
        app.logger.info(request_model)
        return similar_molecules(request_model).json()


@app.route("/target", methods=["POST"])
def target():
    if request.json:
        request_model = LabelMeasurement(**request.json)
        app.logger.info(request_model)
        l = create_label_measurement(request_model)
        return l.json()


@app.route("/api/stats/interaction", methods=["POST"])
def interaction():
    if request.json:
        request_model = InteractionEvent(**request.json)
        app.logger.info(request_model)
        e = create_interaction_event(request_model)
        return e.json()


@app.route("/internal/cache_info")
def cache_info():
    return cache_stats()


@app.route("/internal/clear_cache")
def cache_cleanup():
    return str(clear_cache())


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8081)
else:
    gunicorn_error_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(gunicorn_error_logger.level)
