from flask import Flask, render_template, request, jsonify
from assistant import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.json["message"]
    bot_response = get_response(user_message)
    return jsonify(bot_response)

if __name__ == "__main__":
    app.run(debug=True)
