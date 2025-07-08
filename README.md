# AI Assistant (Chatbot)

This is a simple AI chatbot built using PyTorch and NLTK. It uses intent classification with a basic neural network to respond to user inputs.

---

## 🔧 Setup Instructions

### 1. Clone or download the project
git clone 
cd AI_Assistant

### 2. Install dependencies
pip install numpy torch nltk

### 3. Train the model
python train_model.py

### 4. Run the chatbot
python assistant.py

Type quit to exit.

---

## 📁 Files Overview

- intents.json: Contains training data (intents, patterns, responses)
- train_model.py: Trains the model
- assistant.py: Starts the chatbot
- nltk_utils.py: Tokenization, stemming, bag-of-words logic
- model.py: Defines the neural network
- data.pth: Trained model weights
- app.py: flask app

---

## Features

- Basic intent classification using feedforward neural network
- NLTK-based tokenization and stemming
- Responds to greetings, date/time questions, app/website commands
- File search and opening (interactive prompt)



