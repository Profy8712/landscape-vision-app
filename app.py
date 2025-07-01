from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'landscape_classifier.h5')
model = load_model(MODEL_PATH)
CLASS_NAMES = ['desert', 'green_area', 'cloudy', 'water']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files["file"]
        if img_file:
            img_path = os.path.join("static", img_file.filename)
            img_file.save(img_path)
            img = image.load_img(img_path, target_size=(128, 128))
            x = image.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)[0]
            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx]) * 100
            return render_template(
                "result.html",
                filename=img_file.filename,
                pred_class=CLASS_NAMES[class_idx],
                confidence=confidence
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
