import base64
import time

import flask
from flask import Flask, render_template, request, session
import os
import json
import sun
import config
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.secret_key = b"arefqervqwerfcqwef qwrefewrfqwefq wr w"


def gc():
    total = 0
    files = os.listdir(config.cachedir)
    for file in files:
        if os.path.getmtime(f"{config.cachedir}/{file}") < time.time() - 3600:
            os.remove(f"{config.cachedir}/{file}")
            total += 1
    print(f"Garbage collector removed {total} files")


sched = BackgroundScheduler(daemon=True)
sched.add_job(gc, "interval", hours=1)
sched.start()


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return render_template("index.html", results=None)
    elif request.method == "POST":
        if "image" in request.files:
            pts = json.loads(request.form["points"])

            params = {
                "interactive": False,
                "selected_points": pts,
            }

            if "latitude" in request.form:
                params["latitude"] = float(request.form["latitude"])
            if "longitude" in request.form:
                params["longitude"] = float(request.form["longitude"])

            if "locked_a_angle" in request.form and request.form["locked_a_angle"]:
                params["zenith"] = float(request.form["locked_a_angle"])
            if "locked_b_angle" in request.form and request.form["locked_b_angle"]:
                params["azimuth"] = float(request.form["locked_b_angle"])

            results = sun.process_image(request.files["image"], params)
            print(session)
            if "files" not in session:
                session["files"] = results["hash"]
            elif results["hash"] not in session["files"].split(","):
                session["files"] += "," + results["hash"]
            with open(f"{config.cachedir}/data{results['hash']}.json", "w") as f:
                json.dump(results, f)
            return render_template("index.html", results=json.dumps(results), hash=results["hash"])


@app.route("/results/<hash>")
def results(hash):
    if "files" not in session or hash not in session["files"].split(","):
        flask.abort(403)
    hr = hash.replace("/", "")
    url = f"{config.cachedir}/final{hr}.jpg"
    try:
        with open(url, "rb") as f:
            cont = f.read()
        # os.remove(url)
        return flask.Response(cont, mimetype="image/jpeg")
    except FileNotFoundError:
        flask.abort(404)


@app.route("/data/<hash>")
def data(hash):
    if "files" not in session or hash not in session["files"].split(","):
        flask.abort(403)
    hr = hash.replace("/", "")
    url = f"{config.cachedir}/data{hr}.json"
    try:
        with open(url, "rb") as f:
            cont = f.read()
        # os.remove(url)
        return flask.Response(cont, mimetype="application/json")
    except FileNotFoundError:
        flask.abort(404)


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
