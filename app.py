import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from flask import Flask, request, jsonify

# Load model from the models/ directory
MODEL_PATH = "models/paddy_disease_classifier.pth"
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Label mapping (modify according to your classes)
class_names = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut", "Healthy"]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))
        image = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            result = class_names[predicted.item()]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Paddy Disease Classifier API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
