from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import tensorflow
import base64
import pickle
import numpy as np
import gdown
from PIL import Image
from io import BytesIO
import os
app = Flask(__name__)


file_id = "1kZnG8fyuuEXJ4JHId6g76_KqudrsyqyE"
file_path = "PLANT.pkl"

if not os.path.exists(file_path):
    download_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(download_url, file_path, quiet=False)
with open('./PLANT.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/", methods=["GET"])
def get():
    return render_template('image.html')

@app.route("/", methods=["POST"])
def post():
    data = request.get_json()
    Text_Img = data["TextImg"]

    if Text_Img.startswith('data:image'):
        Text_Img = Text_Img.split(',')[1]
    
    try:
        image_bytes = base64.b64decode(Text_Img)
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('RGB') 
        image = image.resize((224, 224))

        image = np.array(image)
        image = image / 255.0
        image = image.reshape(1, 224, 224, 3)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        
        return jsonify({"prediction": int(predicted_class)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
