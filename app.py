print("Hello, World!")

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import google.generativeai as genai

print("All libraries imported successfully")

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyDlRx9W5_-X2wkiiDmTJNMbHZVuI-KKYM0")

# Load the trained PyTorch model
print("Loading model...")
MODEL_PATH = "./plant-disease-model-complete.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)  # Load model on the correct device
model.eval()  # Set model to evaluation mode

print("Model loaded successfully")
# Define the image transformations (modify as per your training pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match training input size
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

print("Transformations defined successfully")

# Define class labels based on your dataset
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


def check_if_plant(image):
    """Checks if an image contains a plant using Gemini AI"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = "Analyze the given image and determine whether it contains a plant or not. Respond with 'plant' or 'not a plant' only."

    # Convert bytes to a PIL Image
    image_pil = Image.open(io.BytesIO(image.read()))

    response = model.generate_content([prompt, image_pil])
    return response.text.strip().lower() == "plant"


def process_plant(image):
    """Processes the plant image using the ResNet model to classify disease"""
    # Convert image to PIL format
    image_pil = Image.open(io.BytesIO(image.read())).convert("RGB")
    
    # Apply transformations
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # Add batch dimension and send to device
    
    # Run model inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)  # Get probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Get class index

    return {
        "status": "processed",
        "predicted_class": class_labels[predicted_class],  # Convert index to class label
        "confidence": round(probabilities[0, predicted_class].item(), 4)  # Confidence score
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files["image"]

    if check_if_plant(image):
        result = process_plant(image)
        return jsonify(result)
    else:
        return jsonify({"message": "Not a plant"})

if __name__ == "__main__":
    app.run(debug=True)
