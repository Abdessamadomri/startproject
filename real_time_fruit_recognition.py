import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import json
from PIL import Image

# 📦 Charger les classes
with open('classes.json', 'r') as f:
    class_names = json.load(f)

# 📱 Détection de device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.DEFAULT

# 🧠 Charger le modèle pré-entraîné + adapter pour nos classes
model = efficientnet_b0(weights=weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load('efficientnet_fruit_model.pt', map_location=device))
model = model.to(device)
model.eval()

# 📐 Transformation de l’image
transform = weights.transforms()

# 📷 Capture webcam (avec backend DirectShow)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Impossible d'accéder à la webcam.")
    exit()

print("🎥 Webcam démarrée. Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Échec de la lecture vidéo.")
        break

    # Zone centrale de l'image
    h, w, _ = frame.shape
    size = min(h, w) // 2
    x1 = w//2 - size//2
    y1 = h//2 - size//2
    crop = frame[y1:y1+size, x1:x1+size]

    # Préparer l’image
    img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 🔍 Prédiction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        label = class_names[pred.item()] if conf.item() > 0.5 else "غير واضح"

    # 🖼️ Affichage
    cv2.putText(frame, f"{label} ({conf.item()*100:.1f}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), (255, 0, 0), 2)
    cv2.imshow("📸 Prédiction Fruit - Appuie sur q pour quitter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
