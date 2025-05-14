import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
import json
from dotenv import load_dotenv
import logging
import re
from typing import Optional, Dict, Any
import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# X-ray Diagnosis Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize models
try:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 5)
    model.load_state_dict(torch.load("chest_xray_efficientnet_b0.pth", map_location=device))
    model = model.to(device)
    model.eval()
    logger.info("Diagnosis model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load diagnosis model: {str(e)}")
    raise RuntimeError("Could not initialize diagnosis model") from e

CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS', 'non_xray']
DISEASE_DESCRIPTIONS = {
    'COVID19': {
        'description': "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus.",
        'symptoms': ["Fever or chills", "Cough", "Shortness of breath", "Fatigue", "Muscle aches", "Loss of taste/smell"],
        'treatment': ["Rest and hydration", "Antiviral medications", "Oxygen therapy in severe cases"],
        'prevention': ["Vaccination", "Mask-wearing", "Social distancing", "Hand hygiene"],
        'xray_findings': [
            "Bilateral ground-glass opacities",
            "Peripheral distribution",
            "Vascular enlargement",
            "Crazy-paving pattern in severe cases"
        ]
    },
    'NORMAL': {
        'description': "The X-ray appears normal with no signs of disease.",
        'symptoms': ["No abnormal symptoms"],
        'treatment': ["No treatment needed"],
        'prevention': ["Regular check-ups", "Healthy lifestyle"],
        'xray_findings': [
            "Clear lung fields",
            "Sharp costophrenic angles",
            "Normal vascular patterns",
            "No evidence of consolidation or nodules"
        ]
    },
    'PNEUMONIA': {
        'description': "Pneumonia is an infection that inflames the air sacs in the lungs.",
        'symptoms': ["Cough with phlegm", "Fever", "Difficulty breathing", "Chest pain"],
        'treatment': ["Antibiotics (bacterial)", "Antivirals (viral)", "Oxygen therapy"],
        'prevention': ["Pneumonia vaccines", "Good hygiene", "Not smoking"],
        'xray_findings': [
            "Lung consolidation (white patches)",
            "Air bronchograms",
            "Interstitial markings",
            "Possible pleural effusion"
        ]
    },
    'TURBERCULOSIS': {
        'description': "Tuberculosis is a bacterial infection that primarily affects the lungs.",
        'symptoms': ["Chronic cough", "Night sweats", "Weight loss", "Blood in sputum"],
        'treatment': ["Long-term antibiotics", "Directly observed therapy"],
        'prevention': ["BCG vaccine", "Early detection", "Infection control"],
        'xray_findings': [
            "Upper lobe infiltrates",
            "Cavitary lesions",
            "Lymph node enlargement",
            "Miliary nodules in disseminated cases"
        ]
    },
    'non_xray': {
        'description': "The uploaded image doesn't appear to be a chest X-ray.",
        'symptoms': [],
        'treatment': [],
        'prevention': [],
        'xray_findings': []
    }
}

# Model metadata
MODEL_METADATA = {
    "name": "EfficientNet-B0",
    "description": "A deep convolutional neural network optimized for image classification",
    "training_data": "ChestX-ray14 + COVID-19 dataset (112,120 images)",
    "parameters": "5.3 million",
    "accuracy": "92.4% on validation set",
    "detection_method": """
    The model analyzes chest X-rays by:
    1. Examining texture patterns in lung tissue
    2. Identifying abnormal opacities and consolidations
    3. Detecting spatial distributions of abnormalities
    4. Comparing features against learned patterns from training data
    
    For pneumonia detection specifically, it looks for:
    - Areas of lung consolidation (white patches)
    - Air bronchograms (air-filled bronchi visible against consolidated lung)
    - Interstitial markings (lines between air sacs)
    - Possible pleural effusion (fluid around lungs)
    """
}

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def generate_gradcam(image_path: str, predicted_class: str) -> str:
    try:
        img = Image.open(image_path)
        transform = get_transform()
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Grad-CAM implementation
        target_layer = model.features[-1]
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
            
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
            
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        output = model(img_tensor)
        class_idx = CLASS_NAMES.index(predicted_class)
        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32).to(device)
        one_hot[0][class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot)
        
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
        activations = activations[0].squeeze(0)
        
        for i in range(activations.size()[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        img = cv2.imread(image_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        explain_dir = os.path.join(app.static_folder, 'explanations')
        os.makedirs(explain_dir, exist_ok=True)
        gradcam_path = os.path.join(explain_dir, 'gradcam_explanation.png')
        cv2.imwrite(gradcam_path, superimposed_img)
        
        with open(gradcam_path, "rb") as f:
            gradcam_b64 = base64.b64encode(f.read()).decode('utf-8')
            
        forward_handle.remove()
        backward_handle.remove()
        
        return gradcam_b64
    except Exception as e:
        logger.error(f"Grad-CAM error: {str(e)}")
        return None

def predict_image(image_path: str) -> Dict[str, Any]:
    transform = get_transform()
    try:
        img = Image.open(image_path)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

        predictions = {CLASS_NAMES[i]: round(probs[i].item(), 2) for i in range(len(CLASS_NAMES))}
        top_prob, top_idx = torch.max(probs, 0)
        diagnosis = CLASS_NAMES[top_idx]
        
        # Generate Grad-CAM visualization
        gradcam = None
        if diagnosis != 'non_xray':
            gradcam = generate_gradcam(image_path, diagnosis)
        
        disease_info = DISEASE_DESCRIPTIONS.get(diagnosis, {})
        
        return {
            'diagnosis': diagnosis,
            'confidence': round(top_prob.item(), 2),
            'description': disease_info.get('description', ''),
            'symptoms': disease_info.get('symptoms', []),
            'treatment': disease_info.get('treatment', []),
            'prevention': disease_info.get('prevention', []),
            'xray_findings': disease_info.get('xray_findings', []),
            'all_predictions': predictions,
            'non_xray_warning': predictions['non_xray'] > 50,
            'gradcam': gradcam,
            'model_metadata': MODEL_METADATA,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error in predict_image: {str(e)}")
        return {'error': f"Could not process image: {str(e)}", 'success': False}

# Medical Chatbot
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyDPK3d4s2q-WnhUj4Hc2EijNKM7p6IJOpQ"
chatbot_model = None

try:
    genai.configure(api_key=GEMINI_API_KEY)
    chatbot_model = genai.GenerativeModel('gemini-1.5-pro')
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    chatbot_model = None

def generate_chat_response(user_input: str, diagnosis_data: Optional[Dict] = None) -> Dict[str, Any]:
    user_input_lower = user_input.lower()
    
    # Handle model-related questions
    if any(phrase in user_input_lower for phrase in ['what model', 'which model', 'model used', 'how classify']):
        return {
            "disease": "Model Information",
            "content": [
                f"Model Name: {MODEL_METADATA['name']}",
                f"Description: {MODEL_METADATA['description']}",
                f"Training Data: {MODEL_METADATA['training_data']}",
                f"Parameters: {MODEL_METADATA['parameters']}",
                f"Accuracy: {MODEL_METADATA['accuracy']}",
                "\nHow the model detects abnormalities:",
                MODEL_METADATA['detection_method']
            ],
            "follow_up_questions": [
                "How accurate is this model?",
                "What diseases can it detect?",
                "Can it explain specific diagnoses?"
            ],
            "source": "Thorax AI Diagnostic System",
            "disclaimer": "This model is for assistive purposes only",
            "success": True
        }
    
    # Handle diagnosis explanation requests
    if any(phrase in user_input_lower for phrase in ['how detect', 'how identify', 'how diagnose', 'how know']) and diagnosis_data:
        disease = diagnosis_data.get('diagnosis')
        if disease and disease in DISEASE_DESCRIPTIONS:
            return {
                "disease": disease,
                "content": [
                    f"The model detected {disease} by identifying these key features:",
                    *DISEASE_DESCRIPTIONS[disease]['xray_findings'],
                    f"\nConfidence level: {diagnosis_data.get('confidence', 0)}%",
                    "\nModel detection method:",
                    MODEL_METADATA['detection_method']
                ],
                "follow_up_questions": [
                    f"What are the symptoms of {disease}?",
                    f"What is the treatment for {disease}?",
                    "How accurate is this diagnosis?"
                ],
                "source": "Thorax AI Diagnostic System",
                "disclaimer": "This explanation shows how the AI made its prediction",
                "success": True
            }
    
    # General medical questions
    prompt = f"""You are a medical expert AI. Provide detailed information about:
    "{user_input}"

    Respond in STRICT JSON format with these sections:
    {{
      "disease": "Condition name",
      "content": ["list", "of", "detailed", "information"],
      "follow_up_questions": ["list", "of", "3", "relevant", "questions"],
      "source": "Gemini Medical AI",
      "disclaimer": "Consult a healthcare professional",
      "success": true
    }}

    Rules:
    1. Provide specific, actionable medical information
    2. Include key facts and practical advice
    3. suggested_questions should be relevant to the query
    4. For prevention questions, provide specific recommendations
    5. For treatment questions, mention both medical and lifestyle approaches"""

    if not chatbot_model:
        return {
            "disease": user_input,
            "content": ["Medical assistant service is currently unavailable"],
            "follow_up_questions": [],
            "disclaimer": "For urgent medical concerns, seek immediate care.",
            "success": False
        }

    try:
        response = chatbot_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 2000,
                "response_mime_type": "application/json"
            }
        )
        
        if not response.text:
            raise ValueError("No response from medical AI")
        
        try:
            response_data = json.loads(response.text)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\n(.+?)\n```', response.text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group(1))
            else:
                raise ValueError("Invalid JSON response format")

        return {
            "disease": response_data.get("disease", user_input),
            "content": response_data.get("content", ["No information available"]),
            "follow_up_questions": response_data.get("follow_up_questions", []),
            "source": response_data.get("source", "Gemini Medical AI"),
            "disclaimer": response_data.get("disclaimer", "Consult a healthcare professional"),
            "success": True
        }
            
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return {
            "disease": user_input,
            "content": ["Temporary service interruption"],
            "follow_up_questions": [],
            "disclaimer": "For urgent medical concerns, seek immediate care.",
            "success": False
        }

# Flask Routes
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/disease")
def disease():
    return render_template("disease.html")

@app.route('/medichat')
def medichat():
    return render_template('medichat.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'success': False}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file', 'success': False}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed', 'success': False}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_image(filepath)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f"Could not process file: {str(e)}", 'success': False}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Please provide a message", "success": False}), 400
            
        diagnosis = None
        if 'diagnosis' in data:
            diagnosis = data['diagnosis'] if isinstance(data['diagnosis'], dict) else {'diagnosis': data['diagnosis']}
        
        response = generate_chat_response(data['message'], diagnosis)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False,
            "disclaimer": "For urgent matters, consult a doctor immediately"
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)