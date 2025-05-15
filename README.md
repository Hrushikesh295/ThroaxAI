Here's a detailed, structured `README.md` for **Thorax AI**, following the exact format you've requested:

---

# 🩺 Thorax AI: Chest X-Ray Disease Detection + Medical Chatbot

---

## 📖 Project Description

**Thorax AI** is an intelligent, AI-powered web application designed to assist in the early detection of thoracic diseases using chest X-ray images and to provide instant medical guidance via an integrated AI chatbot.

Built with **Python and Streamlit** and integrated with **Google's Gemini API**, the platform allows users to upload chest X-rays, receive disease predictions through a deep learning model (EfficientNet-B0), and interact with a **study-style medical chatbot** trained to answer health-related queries.

### ✨ Key Capabilities:

* Chest X-ray image classification (Pneumonia, Tuberculosis, COVID-19, etc.)
* Real-time heatmap visualizations using Grad-CAM
* Gemini API-powered medical chatbot
* User-friendly web interface with seamless interaction
* Lightweight and fast model inference, suitable for low-resource systems

---

## ⚙️ Installation Instructions

### ✅ Prerequisites

* Python 3.8 or above
* Git (for cloning the repository)
* A Google Gemini API Key
* Virtual environment tool (optional but recommended)

### 🧰 Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/thorax-ai.git
   cd thorax-ai
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the root directory:

   ```
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

---

## ▶️ Usage

### 🔄 Running the App

```bash
streamlit run app.py
```

This will start the app on `http://localhost:8501`.

### 🧪 Interacting with the App

* **Upload Chest X-Ray:** Users can upload an image (JPG/PNG) of their chest X-ray.
* **AI Diagnosis:** The app predicts diseases using EfficientNet-B0 and highlights affected regions using Grad-CAM.
* **Medical Chatbot:** Users can type health-related questions (e.g., “What is Pneumonia?”), and the chatbot provides a medically accurate response powered by Gemini.
* **Live Feedback:** All results and responses are shown in real-time without needing to reload the page.

---

## 🎯 Features

* ✅ **Chest X-Ray Disease Classification** using EfficientNet-B0
* 🔥 **Grad-CAM Heatmaps** to visualize affected lung regions
* 🤖 **Gemini AI Chatbot** for interactive medical Q\&A
* 📁 **Secure Image Upload** and analysis
* 🖥️ **Streamlit-based Web Interface** for seamless UX
* 📡 **Real-time Prediction and Chat Feedback**
* 🧩 **Modular Design** for scalability and easy integration

---

## 💻 Technologies Used

| Category             | Technology / Tool                  |
| -------------------- | ---------------------------------- |
| Language             | Python 3.8+                        |
| Framework            | Streamlit, Flask (internal API)    |
| Deep Learning        | PyTorch, TorchVision               |
| Vision Model         | EfficientNet-B0                    |
| Image Processing     | OpenCV, PIL, Matplotlib            |
| NLP Chatbot          | Gemini API (`google.generativeai`) |
| Visualization        | Grad-CAM                           |
| Environment Handling | python-dotenv                      |

---

## 🤝 Contributing

We welcome contributions!

### 📌 How to Contribute:

* Fork this repository
* Create a new branch (`git checkout -b feature-name`)
* Commit your changes (`git commit -m "Add feature"`)
* Push to the branch (`git push origin feature-name`)
* Open a Pull Request

### 🐛 Reporting Issues:

* Submit bugs or issues via GitHub [Issues](https://github.com/your-username/thorax-ai/issues)
* Include screenshots and steps to reproduce if possible

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

* [EfficientNet-B0 Paper](https://arxiv.org/abs/1905.11946) – For the base model
* [Google Gemini API](https://ai.google.dev/) – For medical chatbot integration
* [Grad-CAM](https://arxiv.org/abs/1610.02391) – For interpretability
* [PyTorch](https://pytorch.org/) – Model training and inference
* [Streamlit](https://streamlit.io/) – Interactive web UI

---

## 📬 Contact Information

**Authors:**

* R. Hrushikesh – 321506402295
* K. Sai Charan – 321506402150
* K. Sai Lokesh – 321506402153
* M. Karthik – 321506402205

**Guide:** Prof. V. Valli Kumari
**Institution:** Andhra University College of Engineering (A), Visakhapatnam

📧 **For queries or collaboration:**
Email: `rameshsrinivasan300@gmail.com` *(Replace with actual contact)*

---
