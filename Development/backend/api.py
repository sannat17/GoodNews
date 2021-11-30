import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.get("/")
def home():
    return "In development"

if __name__ == "__main__":
    app.run()