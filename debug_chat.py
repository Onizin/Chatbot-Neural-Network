import random
import json
import torch
from model import NeuralNet
from nltk_util import bag_of_words, tokenize

print("1. Imports successful")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("2. Device:", device)

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
print("3. Intents loaded")

FILE = "data.pth"
data = torch.load(FILE)
print("4. Data loaded")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
print("5. Data extracted")

model = NeuralNet(input_size, hidden_size, output_size).to(device)
print("6. Model created")

model.load_state_dict(model_state)
print("7. Model state loaded")

model.eval()
print("8. Model set to eval mode")

def get_response_debug(msg):
    print(f"9. Processing message: {msg}")
    sentence = tokenize(msg)
    print(f"10. Tokenized: {sentence}")
    
    X = bag_of_words(sentence, all_words)
    print(f"11. Bag of words shape: {X.shape}")
    
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    print(f"12. Tensor shape: {X.shape}")

    output = model(X)
    print(f"13. Model output: {output}")
    
    _, predicted = torch.max(output, dim=1)
    print(f"14. Predicted: {predicted}")

    tag = tags[predicted.item()]
    print(f"15. Tag: {tag}")
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(f"16. Probability: {prob.item()}")
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"17. Response: {response}")
                return response
    
    response = "Maaf saya tidak mengerti, coba pertanyaan lain."
    print(f"17. Default response: {response}")
    return response

if __name__ == "__main__":
    print("Testing get_response_debug...")
    result = get_response_debug("hello")
    print("Final result:", result)
