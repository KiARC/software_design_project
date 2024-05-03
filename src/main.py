from flask import Flask, render_template, request
import os
import json
import sun
from werkzeug.middleware.profiler import ProfilerMiddleware

app = Flask(__name__)
app.config["PROFILE"] = True
app.wsgi_app = ProfilerMiddleware(
    app.wsgi_app,
    restrictions=[40, "sun"],
    profile_dir="profile",
    filename_format="{time:.0f}-{method}-{path}-{elapsed:.0f}ms.prof",
)


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
            return render_template("index.html", results=json.dumps(results, indent=4))


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
