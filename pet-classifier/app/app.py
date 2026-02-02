# app/app.py
from flask import Flask, render_template, request
import os
from model import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None
    if request.method == "POST":
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            prediction = predict_image(filename)
    return render_template("index.html", prediction=prediction, filename=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
