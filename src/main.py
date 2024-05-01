from flask import Flask, render_template, request
import os
import json
import sun

app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/results", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        print(request.files)
        if "filename" in request.files:
            print(request.files["filename"].read()[:30])
    return render_template('results.html')


@app.route("/test", methods=["GET", "POST"])
def upload_test():
    if request.method == "POST":
        print(request.files)
        if "filename" in request.files:
            print(request.files["filename"].read()[:30])
            pts = json.loads(request.form["points"])
            results = sun.process_image(request.files["filename"], {
                "interactive": False,
                "latitude": 39.277895,
                "longitude": -77.216466,
                "selected_points": pts
            })
            return render_template("upload_frontend.html", results=json.dumps(results, indent=4))
    return render_template('upload_frontend.html')
    

if __name__ == '__main__':
    app.run("0.0.0.0", 5000)