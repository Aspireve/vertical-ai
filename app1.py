print("Hello, World!")

from flask import Flask, request, jsonify
from PIL import Image
import io
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class_labels = [
    "Tomato - Late Blight", "Tomato - Healthy", "Grape - Healthy", "Orange - Citrus Greening",
    "Soybean - Healthy", "Squash - Powdery Mildew", "Potato - Healthy", "Corn - Northern Leaf Blight",
    "Tomato - Early Blight", "Tomato - Septoria Leaf Spot", "Corn - Gray Leaf Spot", "Strawberry - Leaf Scorch",
    "Peach - Healthy", "Apple - Apple Scab", "Tomato - Yellow Leaf Curl Virus", "Tomato - Bacterial Spot",
    "Apple - Black Rot", "Blueberry - Healthy", "Cherry - Powdery Mildew", "Peach - Bacterial Spot",
    "Apple - Cedar Apple Rust", "Tomato - Target Spot", "Pepper Bell - Healthy", "Grape - Leaf Blight",
    "Potato - Late Blight", "Tomato - Mosaic Virus", "Strawberry - Healthy", "Apple - Healthy",
    "Grape - Black Rot", "Potato - Early Blight", "Cherry - Healthy", "Corn - Common Rust",
    "Grape - Black Measles", "Raspberry - Healthy", "Tomato - Leaf Mold", "Tomato - Spider Mites",
    "Pepper Bell - Bacterial Spot", "Corn - Healthy"
]

def check_if_plant(image_pil):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = "Analyze the given image and determine whether it contains a plant or not. Respond with 'plant' or 'not a plant' only."
    response = model.generate_content([prompt, image_pil])
    return response.text.strip().lower() == "plant"

def process_plant(image_pil):
    model = genai.GenerativeModel("gemini-1.5-flash")    
    prompt =  "This is an image of a plant leaf. Identify the disease based on the given possible classes:\n\n" + ", ".join(class_labels) + "\n\nIf the plant is healthy, specify the healthy label from the list. Respond with class label only."
    response = model.generate_content([prompt, image_pil])
    return {"status": "processed", "predicted_class": response.text.strip()}

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files["image"]

    image_pil = Image.open(io.BytesIO(image.read()))

    if check_if_plant(image_pil):
        result = process_plant(image_pil)
        return jsonify(result)
    else:
        return jsonify({"message": "Not a plant"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
