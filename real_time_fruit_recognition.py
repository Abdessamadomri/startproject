import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import json
from PIL import Image, ImageTk
import tkinter as tk

# تحميل أسماء الفواكه
with open('classes.json', 'r') as f:
    class_names = json.load(f)

fruit_calories = {
    "Apple": 52,
    "Banana": 89,
    "Carrot": 41,
    "Grape": 69,
    "Guava": 68,
    "Jujube": 79,
    "Mango": 60,
    "Orange": 47
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load('efficientnet_fruit_model.pt', map_location=device))
model = model.to(device)
model.eval()

transform = weights.transforms()

cap = None
running = False

root = tk.Tk()
root.title("Fruit Recognition + Calories")

label_img = tk.Label(root)
label_img.pack()

label_text = tk.Label(root, text="Resultat: ...", font=("Arial", 18))
label_text.pack(pady=10)

def start_camera():
    global cap, running
    if cap is None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            label_text.config(text="❌ Impossible d'accéder à la webcam.")
            return
    running = True
    update_frame()

def stop_camera():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
        cap = None
    label_img.config(image='')
    label_text.config(text="الكاميرا متوقفة.")

def update_frame():
    global cap, running
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        label_text.config(text="⚠️ فشل قراءة الفيديو.")
        root.after(100, update_frame)
        return

    h, w, _ = frame.shape
    size = min(h, w) // 2
    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2
    crop = frame[y1:y1+size, x1:x1+size]

    img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        label = class_names[pred.item()] if conf.item() > 0.5 else "غير واضح"
        calories = fruit_calories.get(label, "غير معروف")

    # عرض النص فال label (مش على الفيديو)
    label_text.config(text=f"التعرف: {label} ({conf.item()*100:.1f}%) - {calories} kcal")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)

    root.after(10, update_frame)

btn_start = tk.Button(root, text="تشغيل الكاميرا", command=start_camera)
btn_start.pack(side=tk.LEFT, padx=10, pady=10)

btn_stop = tk.Button(root, text="إيقاف الكاميرا", command=stop_camera)
btn_stop.pack(side=tk.RIGHT, padx=10, pady=10)

root.mainloop()
