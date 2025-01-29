#custom model of spam detection
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load dataset
data = pd.read_csv("messages.csv")

# Handle NaN values by filling with an empty string or dropping rows
data['subject'].fillna('', inplace=True)
data['body'].fillna('', inplace=True)

# Combine subject and body
data['text'] = data['subject'] + " " + data['body']

# Label encoding
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])  # primary and spam
category_mapping = {0: 'Primary', 1: 'Spam'}

print(data['label'].value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Convert text into numerical vectors
vectorizer = CountVectorizer(max_features=5000)  # Vectorization (Converting text to numerical data)
X_train_vectors = vectorizer.fit_transform(X_train).toarray()
X_test_vectors = vectorizer.transform(X_test).toarray()

# Dataset class to handle the data for PyTorch
class EmailDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert input data to tensor
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor

    def __len__(self):
        return len(self.labels)  # Returns the total number of samples

    def __getitem__(self, index):
        return self.data[index], self.labels[index]  # Returns a sample and its corresponding label

# Define the model architecture
class EmailClassifier(nn.Module):
    def __init__(self, input_size):
        super(EmailClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(128, 2)  # Output layer for 2 classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize datasets and data loaders
train_dataset = EmailDataset(X_train_vectors, y_train.to_numpy())  # Convert labels to numpy array
test_dataset = EmailDataset(X_test_vectors, y_test.to_numpy())    # Convert labels to numpy array

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_size = X_test_vectors.shape[1]  # Number of features (words)
model = EmailClassifier(input_size)

# Move model to device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # Cross entropy loss function for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training the model
for epoch in range(10):  # Train for 10 epochs
    model.train()  # Set the model to training mode
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update weights

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "email_classifier.pth")
print("Model saved successfully!")

# Flask integration for categorizing new emails
@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.get_json()
        email_text = data.get("subject", "") + " " + data.get("body", "")
        
        # Convert text to vector using the same vectorizer used during training
        email_vector = vectorizer.transform([email_text]).toarray()
        email_tensor = torch.tensor(email_vector, dtype=torch.float32).to(device)

        # Predict category
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to track gradients during inference
            output = model(email_tensor)
            _, predicted = torch.max(output, 1)
            category = category_mapping[predicted.item()]
        
        # Convert to native Python types before returning
        return jsonify({"category": str(category)})  # Ensure category is a string for JSON serialization
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run()


# #crct code
# import os
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # Load dataset
# data = pd.read_csv("messages.csv")

# # Combine subject and body
# data['text'] = data['subject'] + " " + data['body']

# # Label encoding
# label_encoder = LabelEncoder()
# data['label'] = label_encoder.fit_transform(data['label'])  # primary and spam


# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# # Convert text into numerical vectors
# vectorizer = CountVectorizer(max_features=5000)  # Vectorization (Converting text to numerical data)
# X_train_vectors = vectorizer.fit_transform(X_train).toarray()
# X_test_vectors = vectorizer.transform(X_test).toarray()

# # Dataset class to handle the data for PyTorch
# class emaildataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = torch.tensor(data, dtype=torch.float32)  # Convert input data to tensor
#         self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor

#     def __len__(self):
#         return len(self.labels)  # Returns the total number of samples

#     def __getitem__(self, index):
#         return self.data[index], self.labels[index]  # Returns a sample and its corresponding label

# # Define the model architecture
# class EmailClassifier(nn.Module):
#     def __init__(self, input_size):
#         super(EmailClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
#         self.relu = nn.ReLU()  # ReLU activation
#         self.fc2 = nn.Linear(128, 2)  # Output layer for 2 classes

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# # Initialize datasets and data loaders
# train_dataset = emaildataset(X_train_vectors, y_train.to_numpy())  # Convert labels to numpy array
# test_dataset = emaildataset(X_test_vectors, y_test.to_numpy())    # Convert labels to numpy array

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Initialize the model
# input_size = X_test_vectors.shape[1]  # Number of features (words)
# model = EmailClassifier(input_size)
# criterion = nn.CrossEntropyLoss()  # Cross entropy loss function for classification
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# # Training the model
# for epoch in range(10):  # Train for 10 epochs
#     model.train()  # Set the model to training mode
#     for data, labels in train_loader:
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()  # Clear gradients
#         loss.backward()  # Backpropagate the loss
#         optimizer.step()  # Update weights

#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # Save the trained model
# torch.save(model.state_dict(), "email_classifier.pth")
# print("Model saved successfully!")

# # Flask integration for categorizing new emails
# @app.route('/categorize', methods=['POST'])
# def categorize():
#     data = request.get_json()
#     email_text = data.get("subject", "") + " " + data.get("body", "")
    
#     # Convert text to vector using the same vectorizer used during training
#     email_vector = vectorizer.transform([email_text]).toarray()
#     email_tensor = torch.tensor(email_vector, dtype=torch.float32)

#     # Predict category
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():  # No need to track gradients during inference
#         output = model(email_tensor)
#         _, predicted = torch.max(output, 1)
#         category = label_encoder.inverse_transform([predicted.item()])[0]
    
#     return jsonify({"category": category})

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)

# import os
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from flask import Flask, request, jsonify

# app = Flask(__name__)
# #load dataset
# data=pd.read_csv("email_categorize.csv")

# #combine subject and body
# data['text']=data['subject']+" "+data['body']

# #label encode cheyya
# label_encoder=LabelEncoder()
# data['label']=label_encoder.fit_transform(data['label'])  #primary-0 and primary-1

# #split into train and test
# X_train,X_test,y_train,y_test=train_test_split(data['text'],data['label'],test_size=0.2, random_state=42)

# #convert text into numerical vectors
# vectorizer=CountVectorizer(max_features=5000) # ithan vector aakunnath
# X_train_vectors=vectorizer.fit_transform(X_train).toarray()
# X_test_vectors=vectorizer.transform(X_test).toarray()
# # y ithepole vectorizer aakathath ath already label aakiknu lke primary-0 and spam-0
 
# #create a model
# class emaildataset(Dataset):
#     def __init__(self,data,labels):
#         self.data=torch.tensor(data,dtype=torch.float32)
#         self.labels = torch.tensor(labels,dtype=torch.long)
#         # self.labels=torch.tensor(labels,dtype=torch.long)

#         def __len__(self):
#             return len(self.labels)  #ethra 0 and ethra1 ind nokan ithink!?
#             #Returns the total number of samples in the dataset (length of labels).
#             #Purpose: Helps the DataLoader know how many samples are in the dataset.,,,,

#         def __getitem__(self,index):
#             return self.data[index],self.labels[index]   

# class EmailClassifier(nn.Module):
#     def __init__(self,input_size):
#         super(EmailClassifier,self).__init__()
#         self.fc1=nn.Linear(input_size,128)    
#         self.relu=nn.ReLU()
#         self.fc2=nn.Linear(128,2)        #output layers 2 classes

#         def forward(self,x):
#             # Defines the forward pass (how data flows through the model).
#             # Steps:
#             # Pass x through the first layer: self.fc1(x).
#             # Apply ReLU activation: self.relu(x).
#             x=self.fc1(x)
#             x=self.relu(x)
#             x=self.fc2(x)
#             return x

# #----train the model
# train_dataset=emaildataset(X_train_vectors,y_train.to_numpy())  #ithil y_train il athin crct aaya primary or spam ndakum
# #y_train = [0, 1, 0, 1, 0]  # 0 = Primary, 1 = Spam
# test_dataset=emaildataset(X_test_vectors,y_test.to_numpy())  
# #data load cheyyanam
# train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
# test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

# #---initialize the model
# input_size=X_test_vectors.shape[1]
# #shape[0]: The number of training samples (emails).
# #shape[1]: The number of features (e.g., words or tokens used in the vectorization process).
# model=EmailClassifier(input_size)
# criterion=nn.CrossEntropyLoss()
# optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

# for epoch in range(10):  # Train for 10 epochs
#     model.train()
#     for data, labels in train_loader:
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# torch.save(model.state_dict(), "email_classifier.pth")
# print("Model saved successfully!")

# # Flask integration
# @app.route('/categorize', methods=['POST'])
# def categorize():
#     data = request.get_json()
#     email_text = data.get("subject", "") + " " + data.get("body", "")
    
#     # Convert text to vector
#     email_vector = vectorizer.transform([email_text]).toarray()
#     email_tensor = torch.tensor(email_vector, dtype=torch.float32)

#     # Predict category
#     model.eval()
#     with torch.no_grad():
#         output = model(email_tensor)
#         _, predicted = torch.max(output, 1)
#         category = label_encoder.inverse_transform([predicted.item()])[0]
    
#     return jsonify({"category": category})

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)















