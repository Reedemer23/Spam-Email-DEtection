import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load CSV and prepare data as in your sklearn pipeline
csv_data_path = 'C:\\Users\\samri\\Downloads\\email_classification.csv'
df = pd.read_csv(csv_data_path)
df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
X = df['email']
y = df['label']

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X).toarray()
print("TF-IDF matrix shape:", X_tfidf.shape)  # Check feature count

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Dynamically obtain input size from TF-IDF features
input_size = X_train_tensor.shape[1]

# Define a PyTorch model with the correct input size
class SpamClassifierPyTorch(nn.Module):
    def __init__(self, input_size):
        super(SpamClassifierPyTorch, self).__init__()
        self.fc = nn.Linear(input_size, 2)  # Output layer for 2 classes

    def forward(self, x):
        return self.fc(x)

# Instantiate the model, define loss and optimizer
model = SpamClassifierPyTorch(input_size=input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the PyTorch model
for epoch in range(20):  # Adjust the number of epochs as needed
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the PyTorch model
torch.save(model.state_dict(), 'spam_classifier_model.pth')
print("Model saved as 'spam_classifier_model.pth'")

# Calculate accuracy on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for testing
    test_output = model(X_test_tensor)
    test_predictions = torch.argmax(test_output, dim=1)  # Get predicted class labels
    accuracy = (test_predictions == y_test_tensor).float().mean().item()  # Calculate accuracy
    print(f"Model accuracy on test set: {accuracy * 100:.2f}%")
