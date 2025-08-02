import torch
import os
from PIL import Image
from io import BytesIO
from torchvision import transforms
from model import PlantDiseaseCNN
import base64


class PlantDiseaseHandler:
    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = ["Early Blight", "Late Blight", "Healthy"]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model_dir = properties.get("model_dir")
        model_pt = "best_model.pth"
        model_path = os.path.join(model_dir, model_pt)

        self.model = PlantDiseaseCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval().to(self.device)

    def preprocess(self, data):
        print("meow")
        print(data)
        images = []
        for row in data:
            image = Image.open(BytesIO(row.get("body") or row.get("data")))
            image = image.convert('RGB')
            image = self.transform(image)
            images.append(image)
        return torch.stack(images).to(self.device)

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def postprocess(self, outputs):
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        return [{
            "class": self.class_names[preds.item()],
            "confidence": conf.item()
        }]

    def handle(self, data, context):
        inputs = self.preprocess(data)
        outputs = self.inference(inputs)
        return self.postprocess(outputs)