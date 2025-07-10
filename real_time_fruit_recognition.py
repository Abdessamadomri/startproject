import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn

# موديل SimpleCNN (بحال ديالك)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحميل الموديل
num_classes = len(os.listdir("Fruit_Dataset"))  # تأكد أن هذا المسار صحيح ويحتوي على فولدرات الفواكه
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load('fruit_classifier_model.pt', map_location=device))
model.eval()

# تحويل الصورة للتحضير للتصنيف
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# أسماء الفواكه (مجلدات داخل Fruit_Dataset)
class_names = sorted(os.listdir("Fruit_Dataset"))

cap = cv2.VideoCapture(0)  # تفعيل الكاميرا

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # قطع مربع وسط الشاشة باش تصنف فقط هاديك المنطقة (يمكن تعديل الإحداثيات)
    x_start, y_start = 100, 100
    x_end, y_end = 380, 380
    crop = frame[y_start:y_end, x_start:x_end]

    img_tensor = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        label = class_names[pred.item()] if conf.item() > 0.5 else "غير واضح"

    # عرض النتيجة على الشاشة مع رسم مربع على منطقة القص
    cv2.putText(frame, f"Prediction: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

    cv2.imshow('Fruit Recognition - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
