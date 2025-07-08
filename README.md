# AI Assistant (Chatbot)

This is a simple AI chatbot built using PyTorch and NLTK. It uses intent classification with a basic neural network to respond to user inputs.

---

## ✨ Features

- Intent classification using a neural network
- Predefined responses for greetings, jokes, and thank yous
- Tells current time and date
- Opens websites (YouTube, Google, Gmail, etc.)
- Opens desktop applications (Notepad, Calculator, CMD, etc.)
- File search capability on Desktop and Downloads folders
- Built using PyTorch, NLTK, and simple rule-based logic

---

## 🔧 Setup Instructions

### 1. Clone or download the project
- git clone https://github.com/Yash6352-rs/AI_Assistant.git
- cd AI_Assistant

### 2. Install dependencies
pip install numpy torch nltk

### 3. Train the model
python train_model.py

### 4. Launch the Web App
python app.py

Then open your browser and go to:
http://127.0.0.1:5000/

Type quit to exit.

---

## 📁 Files Overview

- intents.json: Contains training data (intents, patterns, responses)
- train_model.py: Trains the model
- assistant.py: Starts the chatbot
- nltk_utils.py: Tokenization, stemming, bag-of-words logic
- model.py: Defines the neural network
- data.pth: Trained model weights
- app.py: Flask web server
- index.html: Web UI template
- style.css: Web UI styling

---

## Example Commands

You can ask the assistant things like:

| **Intent**   | **Example Commands**                           |
| ------------ | ---------------------------------------------- |
| Greeting     | `hello`, `good morning`, `hey`                 |
| Time/Date    | `what time is it`, `today's date`              |
| Jokes        | `tell me a joke`, `make me laugh`              |
| File Search  | `find file`, then provide the filename         |
| Open App     | `open notepad`, `start calculator`, `open cmd` |
| Open Website | `open youtube`, `go to gmail`, `open github`   |
| Farewell     | `exit`, `quit`                                 |

---

## Author
Created by Yash
Feel free to connect on LinkedIn or GitHub
