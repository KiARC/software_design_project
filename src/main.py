from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/results", methods=["GET", "POST"])
def result():
    return render_template('results.html')

if __name__ == '__main__':
    app.run("0.0.0.0", 5000)