import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import json
import os

# ✅ Paramètres
batch_size = 32
num_epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️ Appareil utilisé : {device}")

# ✅ Charger les transformations recommandées
weights = EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()

# ✅ Charger le dataset
dataset_path = 'Fruit_Dataset'

if not os.path.exists(dataset_path):
    print(f"❌ Le dossier '{dataset_path}' n'existe pas !")
    exit()

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ✅ Affichage de diagnostic
print(f"📊 Nombre de classes : {len(dataset.classes)}")
print(f"📸 Nombre total d'images : {len(dataset)}")
print("📂 Classes :", dataset.classes)

if len(dataset) == 0:
    print("❌ Aucune image trouvée. Vérifie le contenu de 'Fruit_Dataset'.")
    exit()

# 💾 Sauvegarder les classes
with open('classes.json', 'w') as f:
    json.dump(dataset.classes, f)
print("💾 Fichier classes.json enregistré.")

# 🧠 Charger le modèle EfficientNet
model = efficientnet_b0(weights=weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(dataset.classes))
model = model.to(device)

# ⚙️ Fonction de coût et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 🔁 Entraînement
print("🚀 Début de l'entraînement...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(dataset)
    epoch_acc = running_corrects.double() / len(dataset)
    print(f"📈 Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

# 💾 Sauvegarde du modèle
torch.save(model.state_dict(), 'efficientnet_fruit_model.pt')
print("✅ Modèle entraîné et sauvegardé sous le nom efficientnet_fruit_model.pt")
