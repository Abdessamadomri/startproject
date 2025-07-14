import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import json
import sys

# Vérifier si le chemin de l’image a été donné
if len(sys.argv) != 2:
    print("❗ Utilisation : python predict.py chemin_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# 📦 Charger les classes
with open('classes.json', 'r') as f:
    class_names = json.load(f)

# 🧠 Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Modifier la dernière couche pour s'adapter aux classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load('efficientnet_fruit_model.pt', map_location=device))
model = model.to(device)
model.eval()

# 🔁 Prétraitement de l’image
transform = weights.transforms()
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# 🔍 Prédiction
with torch.no_grad():
    output = model(img_tensor)
    prob = torch.nn.functional.softmax(output, dim=1)
    conf, pred = torch.max(prob, 1)
    predicted_class = class_names[pred.item()]
    confidence = conf.item()

# 🖨️ Résultat
print(f"📷 Image : {image_path}")
print(f"✅ Prédiction : {predicted_class} ({confidence*100:.2f}%)")
