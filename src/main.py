import base64
import pathlib

import flask
from flask import Flask, render_template, request, session
import os
import json
import sun

app = Flask(__name__)
app.secret_key = b"arefqervqwerfcqwef qwrefewrfqwefq wr w"


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

            results = sun.process_image(request.files["image"], params)
            print(session)
            if "files" not in session:
                session["files"] = results["hash"]
            else:
                session["files"] += "," + results["hash"]
            with open(str(pathlib.Path(__file__).parent.parent.joinpath("cache/data{}.json".format(results["hash"]))), "w") as f:
                json.dump(results, f)
            return render_template(
                "index.html", results=json.dumps(results), hash=results["hash"]
            )


@app.route("/results/<hash>")
def results(hash):
    if "files" not in session or hash not in session["files"].split(","):
        flask.abort(403)
    url = pathlib.Path(__file__).parent.parent.joinpath("cache/final{}.jpg".format(hash.replace("/", "")))
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
    url = pathlib.Path(__file__).parent.parent.joinpath("cache/data{}.json".format(hash.replace("/", "")))
    try:
        with open(url, "rb") as f:
            cont = f.read()
        # os.remove(url)
        return flask.Response(cont, mimetype="application/json")
    except FileNotFoundError:
        flask.abort(404)


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
