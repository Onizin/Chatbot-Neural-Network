import json
import random
import numpy as np
from nltk_util import tokennize,stem,bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('D:\\project\\neuro\\Neuro-Linguistic-Processing\\intents.json','r') as f: 
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokennize(pattern)
        all_words.extend(w)
        xy.append((w,tag))




ignore_words = ['?','!',',','.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(all_words)
# print("========================")
# print(tags)

X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label) #cross entropy loss



X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("============================")
print(X_train)
print("============================")
print(Y_train)
print("============================")

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    
    # datasetidx
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
# # hyperparameter
dataset = ChatDataSet()

train_loader = DataLoader(dataset=dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=0)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

#entropyloss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # foward
        outputs = model(words)
        
        loss = criterion(outputs,labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        # Update the running total of correct predictions and samples
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        # output = (outputs>0.5).float()
        # ,Accuracy: { correct/output.shape[0]}
        # correct = (output == labels).sum()
    accuracy = 100 * total_correct / total_samples
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f},Accuracy: { accuracy:.2f}')
        


print(f'final loss: {loss.item():.4f}')

# #=======================================================================
# masih error

# # def check_Accuracy(loader,model):
# #     n_correct =0
# #     n_samples =0
# #     model.eval()
# #     with torch.no_grad():
# #         for (words,labels) in loader:
# #             words = words.to(device)
# #             labels = labels.to(dtype=torch.long).to(device)
# #             outputs = model(words)

# #             #value, index
# #             _, predictions = outputs.max(1)
# #             n_correct += (predictions==labels).sum()
# #             n_samples += predictions.size(0)
            
# #     acc = 100.0 * ((float(n_correct)) / (float(n_samples)))
# #     print(f'accuracy = {acc:.2f}')
# #     model.eval()

# # check_Accuracy(test_loader,model)
# =====================================================================

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')