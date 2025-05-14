document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    
    let currentDiagnosis = null;
    
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') sendMessage();
    });
    
    uploadBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            uploadXRay(this.files[0]);
        }
    });
    
    function sendMessage() {
        const message = chatInput.value.trim();
        if (message === '') return;
        
        displayMessage(message, true);
        chatInput.value = '';
        
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'bot-message';
        typingIndicator.id = 'typing-indicator';
        typingIndicator.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
        chatBox.appendChild(typingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                diagnosis: currentDiagnosis
            })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('typing-indicator')?.remove();
            if (data.success) {
                displayMessage(data);
            } else {
                displayMessage("Sorry, I couldn't process your request. Please try again.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('typing-indicator')?.remove();
            displayMessage("There was an error connecting to the server. Please try again later.");
        });
    }
    
    function uploadXRay(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const uploadingIndicator = document.createElement('div');
        uploadingIndicator.className = 'bot-message';
        uploadingIndicator.textContent = 'Analyzing X-ray...';
        chatBox.appendChild(uploadingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            uploadingIndicator.remove();
            if (data.success) {
                currentDiagnosis = data;
                displayDiagnosisResult(data);
            } else {
                displayMessage(data.error || "Couldn't process the X-ray. Please try another image.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            uploadingIndicator.remove();
            displayMessage("There was an error uploading your X-ray. Please try again.");
        });
    }
    
    function displayMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'user-message' : 'bot-message';
        
        if (typeof message === 'object') {
            let html = `<h4>${message.disease}</h4>`;
            
            if (Array.isArray(message.content)) {
                message.content.forEach(item => {
                    html += `<p>${item}</p>`;
                });
            } else {
                html += `<p>${message.content}</p>`;
            }
            
            if (message.follow_up_questions && message.follow_up_questions.length > 0) {
                html += `<div class="follow-up-questions">
                    <p>Related questions:</p>
                    <ul>`;
                message.follow_up_questions.forEach(question => {
                    html += `<li><a href="#" class="follow-up" onclick="sendFollowUp('${question}')">${question}</a></li>`;
                });
                html += `</ul></div>`;
            }
            
            html += `<div class="disclaimer"><small>${message.disclaimer}</small></div>`;
            messageDiv.innerHTML = html;
        } else {
            messageDiv.textContent = message;
        }
        
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    
    function displayDiagnosisResult(data) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'bot-message diagnosis-result';
        
        let html = `<h4>Diagnosis Result</h4>
            <p><strong>Condition:</strong> ${data.diagnosis}</p>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
            <div class="confidence-meter">
                <div class="confidence-level" style="width: ${data.confidence}%"></div>
            </div>
            <p>${data.description}</p>
            <div class="model-info">
                <h5>Model Used: ${data.model_metadata.name}</h5>
                <p>${data.model_metadata.description}</p>
            </div>`;
        
        if (data.symptoms && data.symptoms.length > 0) {
            html += `<div class="info-section">
                <h5>Symptoms:</h5>
                <ul>`;
            data.symptoms.forEach(symptom => {
                html += `<li>${symptom}</li>`;
            });
            html += `</ul></div>`;
        }
        
        if (data.treatment && data.treatment.length > 0) {
            html += `<div class="info-section">
                <h5>Treatment:</h5>
                <ul>`;
            data.treatment.forEach(treatment => {
                html += `<li>${treatment}</li>`;
            });
            html += `</ul></div>`;
        }
        
        if (data.prevention && data.prevention.length > 0) {
            html += `<div class="info-section">
                <h5>Prevention:</h5>
                <ul>`;
            data.prevention.forEach(prevention => {
                html += `<li>${prevention}</li>`;
            });
            html += `</ul></div>`;
        }
        
        if (data.xray_findings && data.xray_findings.length > 0) {
            html += `<div class="info-section">
                <h5>X-ray Findings:</h5>
                <ul>`;
            data.xray_findings.forEach(finding => {
                html += `<li>${finding}</li>`;
            });
            html += `</ul></div>`;
        }
        
        if (data.gradcam) {
            html += `<div class="info-section">
                <h5>Model Attention Heatmap:</h5>
                <p>Red/yellow areas show where the model focused most for diagnosis</p>
                <img src="data:image/png;base64,${data.gradcam}" class="heatmap-image" alt="Model attention heatmap">
            </div>`;
        }
        
        if (data.non_xray_warning) {
            html += `<div class="warning"><i class="fas fa-exclamation-triangle"></i> 
                This image may not be a chest X-ray. Results may be inaccurate.</div>`;
        }
        
        html += `<div class="action-buttons">
            <button onclick="explainDiagnosis()" class="explain-btn">
                <i class="fas fa-question-circle"></i> Explain this prediction
            </button>
            <button onclick="showModelDetails()" class="model-btn">
                <i class="fas fa-brain"></i> Show model details
            </button>
        </div>`;
        
        resultDiv.innerHTML = html;
        chatBox.appendChild(resultDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    
    window.sendFollowUp = function(question) {
        document.getElementById('chat-input').value = question;
        sendMessage();
        return false;
    };
    
    window.explainDiagnosis = function() {
        if (!currentDiagnosis) return;
        document.getElementById('chat-input').value = "How did you detect " + currentDiagnosis.diagnosis + "?";
        sendMessage();
    };
    
    window.showModelDetails = function() {
        document.getElementById('chat-input').value = "What model was used for diagnosis?";
        sendMessage();
    };
});

window.sendFollowUp = function(question) {
    document.getElementById('chat-input').value = question;
    document.querySelector('#send-btn').click();
    return false;
};

window.explainDiagnosis = function() {
    document.getElementById('chat-input').value = "Explain this diagnosis";
    document.querySelector('#send-btn').click();
};

window.showModelDetails = function() {
    document.getElementById('chat-input').value = "What model was used for diagnosis?";
    document.querySelector('#send-btn').click();
};