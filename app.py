from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import PlantDiseaseCNN

CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

model = PlantDiseaseCNN(num_classes=len(CLASS_NAMES))

state_dict = torch.load("best_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()

        predicted_class = CLASS_NAMES[predicted_class_idx]

        return {
            "prediction": predicted_class,
            "confidence": float(confidence),
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/ping")
async def health_check():
    return {"status": "healthy", "device": str(device)}


@app.get("/model-info")
async def model_info():
    return {
        "classes": CLASS_NAMES,
        "device": str(device),
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)