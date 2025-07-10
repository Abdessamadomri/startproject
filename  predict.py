import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import json
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self._init_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _init_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 100, 100)
            x = self.pool(torch.relu(self.conv1(dummy_input)))
            x = self.pool(torch.relu(self.conv2(x)))
            self.flattened_size = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('classes.json', 'r') as f:
    class_names = json.load(f)

model = SimpleCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('fruit_classifier_model.pt', map_location=device))
model.eval()

def get_images_grouped_by_folder(base_dir='Fruit_Dataset'):
    data = {}
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue
        rel_folder = os.path.relpath(root, base_dir)
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            data[rel_folder] = [os.path.join(root, img) for img in images]
    return data

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)
    if conf.item() > 0.5:
        return class_names[predicted.item()]
    else:
        return "غير واضح"

root = tk.Tk()
root.title("Fruit Classifier - Grouped View")

tree = ttk.Treeview(root)
tree.pack(expand=True, fill='both')

data = get_images_grouped_by_folder()

for folder, images in data.items():
    folder_id = tree.insert('', 'end', text=folder, open=False)
    for img_path in images:
        img_name = os.path.basename(img_path)
        tree.insert(folder_id, 'end', text=img_name, values=(img_path,))

def on_tree_select(event):
    selected = tree.selection()
    if not selected:
        return
    item = selected[0]
    if tree.parent(item):
        img_path = tree.item(item, 'values')[0]
        predicted_class = predict_image(img_path)
        messagebox.showinfo("نتيجة التصنيف", f"الصورة:\n{tree.item(item, 'text')}\n\nهي من صنف: {predicted_class}")

tree.bind('<<TreeviewSelect>>', on_tree_select)

root.mainloop()
