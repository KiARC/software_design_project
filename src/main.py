import base64

import flask
from flask import Flask, render_template, request
import os
import json
import sun

app = Flask(__name__)


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
            return render_template("index.html", results=json.dumps(results), hash=results["hash"])


@app.route("/results/<hash>")
def results(hash):
    url = "/tmp/final{}.jpg".format(hash.replace("/", ""))
    with open(url, "rb") as f:
        cont = f.read()
    # os.remove(url)
    return flask.Response(cont, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
