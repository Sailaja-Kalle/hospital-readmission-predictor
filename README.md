

```markdown
# 🏥 Hospital Readmission Predictor

An advanced AI-powered web app that predicts 30-day hospital readmission risk using Machine Learning, AI Agents and MCP Servers.

## 🌐 Live Demo
👉 [Click here to try the app](https://hospital-readmission-predictor-project.streamlit.app)

## ✨ Features
- 🤖 LightGBM ML Model with 97% accuracy
- 🧠 Groq AI Agent with clinical reasoning
- 🔌 MCP Server for tool integration
- 📊 Real-time risk prediction
- 💊 Top 3 risk factors identification
- 📋 Recommended actions before discharge
- 🎨 Beautiful gradient UI

## 🛠 Tech Stack
- **Python 3.11**
- **LightGBM** — ML Model (97% accuracy)
- **Groq Llama 3.3** — AI Agent
- **FastMCP** — MCP Server
- **Streamlit** — Web UI
- **LangChain** — Agent framework
- **GitHub + Streamlit Cloud** — Deployment

## 🚀 How to Run Locally

1. Clone this repository
```
git clone https://github.com/Sailaja-Kalle/hospital-readmission-predictor.git
cd hospital-readmission-predictor
```

2. Create virtual environment
```
py -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Add your Groq API key in `.env`
```
GROQ_API_KEY=your_key_here
```

5. Train the model first
```
python model.py
```

6. Run the app
```
streamlit run app.py
```

## 📊 How It Works
1. Doctor enters patient details in the form
2. LightGBM model predicts readmission probability
3. Groq AI Agent analyzes and explains the risk
4. Top 3 risk factors are identified
5. Recommended actions before discharge are suggested

## ⚠ Disclaimer
This app is for **educational purposes only**. Always consult a qualified doctor for medical decisions.

