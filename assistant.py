import random
import json
import torch
import webbrowser
import os
from pathlib import Path
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from datetime import datetime

# Load intents
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)

# Load model and data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Context tracking for multi-turn (file search)
pending_file_search = {"active": False}

def get_response(sentence):
    sentence_tokens = tokenize(sentence)

    # If waiting for filename from user
    if pending_file_search["active"]:
        pending_file_search["active"] = False  # reset state
        filename = sentence.strip().lower()

        search_dirs = [
            Path.home() / "Desktop",
            Path.home() / "Downloads"
        ]
        found_files = []

        for search_dir in search_dirs:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if filename in file.lower():
                        full_path = os.path.join(root, file)
                        found_files.append(full_path)

        if found_files:
            os.startfile(found_files[0])
            return {"response": f"I found and opened the file: {found_files[0]}"}
        else:
            return {"response": "Sorry, I couldn't find that file."}

    X = bag_of_words(sentence_tokens, all_words)
    X = torch.from_numpy(X).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.65:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])

                # Time
                if tag == "time":
                    now = datetime.now().strftime("%H:%M:%S")
                    return {"response": f"The current time is {now}"}

                # Date
                elif tag == "date":
                    today = datetime.now().strftime("%d %B %Y")
                    return {"response": f"Today's date is {today}"}

                # Exit App
                elif tag == "goodbye":
                    return {"response": response, "exit": True}

                # Greetings
                elif tag in ["greeting_general", "greeting_morning", "greeting_afternoon", "greeting_evening", "greeting_night"]:
                    return {"response": response}

                # Open Websites
                elif tag == "open_youtube":
                    webbrowser.open("https://www.youtube.com")
                elif tag == "open_google":
                    webbrowser.open("https://www.google.com")
                elif tag == "open_gmail":
                    webbrowser.open("https://mail.google.com")
                elif tag == "open_maps":
                    webbrowser.open("https://www.google.com/maps")
                elif tag == "open_github":
                    webbrowser.open("https://github.com")
                elif tag == "open_linkedin":
                    webbrowser.open("https://www.linkedin.com")
                elif tag == "open_chatgpt":
                    webbrowser.open("https://chat.openai.com")
                elif tag == "open_whatsapp_web":
                    webbrowser.open("https://web.whatsapp.com")
                elif tag == "open_facebook":
                    webbrowser.open("https://www.facebook.com")
                elif tag == "open_instagram":
                    webbrowser.open("https://www.instagram.com")
                elif tag == "open_amazon":
                    webbrowser.open("https://www.amazon.com")
                elif tag == "open_netflix":
                    webbrowser.open("https://www.netflix.com")
                elif tag == "open_claude":
                    webbrowser.open("https://claude.ai")

                # Open Apps
                elif tag == "open_calculator":
                    os.system("start calc")
                elif tag == "open_cmd":
                    os.system("start cmd")
                elif tag == "open_notepad":
                    os.system("start notepad")
                elif tag == "open_word":
                    os.system("start winword")
                elif tag == "open_excel":
                    os.system("start excel")
                elif tag == "open_powerpoint":
                    os.system("start powerpnt")

                # File Search Trigger
                elif tag == "file_search":
                    pending_file_search["active"] = True
                    return {"response": "What file name are you looking for?"}

                return {"response": response}

    return {"response": "Sorry, I didn’t understand that."}
