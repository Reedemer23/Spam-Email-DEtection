import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Define the RETVec model
class RETVecCSV(nn.Module):
    def __init__(self, num_labels):
        super(RETVecCSV, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(hidden_state)
        return logits, hidden_state

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_retvec_features(email_text, tokenizer, max_length=128):
    encoding = tokenizer.encode_plus(
        email_text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask

def train_retvec_model(csv_file):
    data = pd.read_csv(csv_file)
    emails = data['email'].values
    labels = data['label'].values

    # Encode the labels ('spam' and 'ham')
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(emails, encoded_labels, test_size=0.2, random_state=42)

    # Initialize model, loss function, and optimizer
    retvec_model = RETVecCSV(num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retvec_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(retvec_model.parameters(), lr=1e-5)

    # Training loop
    for epoch in range(5):  # Adjust the number of epochs as needed
        total_loss = 0
        retvec_model.train()
        for email, label in zip(X_train, y_train):
            input_ids, attention_mask = get_retvec_features(email, tokenizer)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)

            # Forward pass
            outputs, _ = retvec_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, label_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(X_train)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    # Save the trained model
    torch.save(retvec_model.state_dict(), 'retvec_model.pth')
    print("Training complete and model saved.")

    return X_test, y_test, retvec_model

def evaluate_retvec_model(X_test, y_test, model):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_predictions = []

    with torch.no_grad():
        for email in X_test:
            # Get features for the email text
            input_ids, attention_mask = get_retvec_features(email, tokenizer)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get the predicted label
            predicted_label = torch.argmax(outputs, dim=1).item()
            all_predictions.append(predicted_label)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, all_predictions)
    print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')
    return accuracy

# Main execution for training and evaluating the model
if __name__ == "__main__":
    csv_file_path = 'C:\\Users\\samri\\Downloads\\email_classification.csv'
    
    # Train the model and get test data
    X_test, y_test, trained_model = train_retvec_model(csv_file_path)

    # Evaluate the trained model's accuracy
    evaluate_retvec_model(X_test, y_test, trained_model)
