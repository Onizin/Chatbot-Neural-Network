from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    try:
        text = request.get_json().get("message")
        print(f"Received message: {text}")
        
        # Import here to catch any errors
        from chat import get_response
        print("Imported get_response successfully")
        
        response = get_response(text)
        print(f"Got response: {response}")
        
        message = {"answer": response}
        return jsonify(message)
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": "Maaf, terjadi error. Silakan coba lagi."}), 500

if __name__ == "__main__":
    app.run(debug=True)
