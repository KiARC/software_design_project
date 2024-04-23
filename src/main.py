import flask

app = flask.Flask(__name__)

@app.route("/")
def main():
    return "Hello world"


if __name__ == '__main__':
    app.run("0.0.0.0", 5000)