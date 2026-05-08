# Capstone Chatbot Setup Guide
This guide will help you set up the project from scratch, including installing dependencies, setting up a virtual environment, and running the chatbot with Ollama + LLaMA 3.1.

# 1. Prerequisites
Make sure you have installed:
* Python (>= 3.9 recommended)
* Git (optional, for cloning repo)

# 2. Clone the Repository
git clone <your-repo-url>
cd Capstone_Chatbot

# 3. Create Virtual Environment (`myenv`)

### Windows

```bash
python -m venv myenv
myenv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv myenv
source myenv/bin/activate
```

---

# 4. Install Project Dependencies
```bash
pip install -r requirements.txt
```

# 5. Install Ollama

### Download & Install

Go to: https://ollama.com/download
Install Ollama for your OS.

---

# 📥 6. Pull LLaMA 3.1 Model

After installing Ollama, run:

```bash
ollama pull llama3.1
```

👉 This downloads the model locally.

---

# ▶️ 7. Run Ollama Server

Start Ollama:

```bash
ollama run llama3.1
```

OR simply ensure Ollama is running in the background.

---

# 8. Troubleshooting

### ❌ Ollama not found

* Ensure Ollama is installed and added to PATH

### ❌ Model not found

```bash
ollama pull llama3.1
```

### ❌ Module errors

```bash
pip install -r requirements.txt
```

### ❌ Slow performance

* Running on CPU → expected
* Use smaller models for faster responses

# 9. Running Frontend

```
Streamlit run frontend.py
```

---

Happy building! 💡
