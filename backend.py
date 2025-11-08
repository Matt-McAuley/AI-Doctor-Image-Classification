from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from scipy import io
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = './models'
DOMAINS = ['Blood_Cancer', 'Bone_Fracture', 'Brain_MRI', 'Breast_Cancer', 'Chest_Xray']
CLASSES_PER_DOMAIN = {
    'Blood_Cancer': ['benign', 'early_pre-b', 'pre-b', 'pro-b'],
    'Bone_Fracture': ['fractured', 'not-fractured'],
    'Brain_MRI': ['giloma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
    'Breast_Cancer': ['idc', 'non_idc'],
    'Chest_Xray': ['normal', 'pneumonia']
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_model(path, num_classes):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

general_model = load_model(os.path.join(MODEL_DIR, 'General_Model.pth'), num_classes=len(DOMAINS))
domain_models = {
    d: load_model(os.path.join(MODEL_DIR, f'{d}_Model.pth'), num_classes=len(CLASSES_PER_DOMAIN[d]))
    for d in DOMAINS
}

def classify_image_file(file_bytes):
    """Classify image bytes from frontend upload."""
    image = Image.open(io.BytesIO(file_bytes)).convert('L')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Step 1: Predict domain
    with torch.no_grad():
        domain_logits = general_model(img_tensor)
        domain_idx = torch.argmax(domain_logits, dim=1).item()
        domain = DOMAINS[domain_idx]

    # Step 2: Predict class within that domain
    model = domain_models[domain]
    with torch.no_grad():
        class_logits = model(img_tensor)
        class_idx = torch.argmax(class_logits, dim=1).item()

    return {
        'domain': domain,
        'domain_index': domain_idx,
        'class_index': class_idx
    }

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    result = classify_image_file(file.read())
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)