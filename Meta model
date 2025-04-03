from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import torch
from transformers import BertTokenizer
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
import bcrypt
from PIL import Image
import pytesseract
from email import policy
from email.parser import BytesParser
import joblib
import pandas as pd
from bs4 import BeautifulSoup
from flask_mysqldb import MySQL
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#absolute modules
from model1 import RETVecCSV  
from model2 import SpamClassifierPyTorch


app = Flask(__name__)
app.secret_key = ''
app.config['API_KEY'] = ''
CORS(app)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'  
app.config['MYSQL_USER'] = 'your_username'  
app.config['MYSQL_PASSWORD'] = 'your_passwd' 
app.config['MYSQL_DB'] = 'your_db'  

mysql = MySQL(app)

def get_user_by_credentials(email, password):
    # Connect to MySQL and execute the query
    cur = mysql.connection.cursor()
    query = "SELECT name FROM user_cl WHERE email = %s AND password = %s"
    cur.execute(query, (email, password))
    user = cur.fetchone()
    cur.close()
    return user
    
# Directory for uploaded email files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Define allowed upload folder and extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'eml', 'docx'}

# Load the RETVec model and tokenizer
retvec_model = RETVecCSV(num_labels=2)  
state_dict = torch.load('retvec_model.pth')
retvec_model.load_state_dict(state_dict)
retvec_model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the Tesseract model
tesseract_model = SpamClassifierPyTorch(input_size=499)
tesseract_model.load_state_dict(torch.load('spam_classifier_model.pth'))  
tesseract_model.eval()

# Load data from CSV
csv_data_path = 'C:\\Users\\samri\\Downloads\\email_classification.csv'
df = pd.read_csv(csv_data_path)

# Prepare X from the email texts
X = df['email']

# Now apply the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(X) 

# Updated classify_with_tesseract function
def classify_with_tesseract(email_text, target_size=499):
    global vectorizer  # Access the global vectorizer
    try:
        # Step 1: Apply TF-IDF transformation
        text_tfidf = vectorizer.transform([email_text]).toarray()  # Convert single text input to array
        input_tensor = torch.tensor(text_tfidf, dtype=torch.float32)  # Convert to PyTorch tensor 

        # Step 2: Generate a simple feature (e.g., length of the email text)
        feature = np.array([len(email_text)])  # Simple feature: text length

        # Step 3: Ensure the feature size matches target_size
        if feature.shape[0] < target_size:
            padding = np.zeros(target_size - feature.shape[0])  # Pad with zeros
            feature = np.concatenate([feature, padding])
        elif feature.shape[0] > target_size:
            feature = feature[:target_size]  # Truncate if necessary

        # Step 4: Perform model inference
        with torch.no_grad():  # Disable gradient calculation
            output = tesseract_model(input_tensor)
            prediction = torch.argmax(output, dim=1)  # Get predicted class index

        # Step 5: Return classification result
        return 'Spam' if prediction.item() == 1 else 'Ham'

    except Exception as e:
        print(f"Error in classify_with_tesseract: {e}")
        return None
    
# Assuming RETVec features have 768 dimensions, and Tesseract features have 501 dimensions
def combine_features(retvec_features, tesseract_feature):
    # Ensure RETVec features are padded to 1266 dimensions
    target_retvec_size = 1266
    if len(retvec_features) < target_retvec_size:
        padding_size = target_retvec_size - len(retvec_features)
        retvec_features = np.pad(retvec_features, (0, padding_size), mode='constant', constant_values=0)
    elif len(retvec_features) > target_retvec_size:
        retvec_features = retvec_features[:target_retvec_size]

    # Ensure Tesseract feature is a single numeric value
    if not isinstance(tesseract_feature, (list, np.ndarray)) or len(tesseract_feature) != 1:
        raise ValueError("Tesseract features must must be a single numeric value.")

    # Concatenate RETVec features (768) and Tesseract feature (1)
    combined_features = np.concatenate((retvec_features, tesseract_feature), axis=0)
    
    # Adjust combined feature size to 1267 (if needed)
    if len(combined_features) > 1267:
        combined_features = combined_features[:1267]

    # Ensure the combined features have exactly 1267 dimensions
    assert len(combined_features) == 1267, f"Combined feature size is {len(combined_features)}, expected 1267."

    return combined_features


# Modify the Random Forest training block to use the combined features
if os.path.exists('random_forest_model.pth'):
    classifier = joblib.load('random_forest_model.pth')  
else:
    print("No Random Forest model found. Initializing with sample data.")
    
    # Sample email data for initial training 
    sample_emails = [
        "This is a spam email with promotional content.",
        "Hey, let's meet for lunch tomorrow.",
    ]
    sample_labels = [1, 0]  # 1 for Spam, 0 for Ham

    # Extract features using RETVec and Tesseract
    retvec_features = [get_retvec_features(email, retvec_model, tokenizer) for email in sample_emails]
    tesseract_predictions = [classify_with_tesseract(email) for email in sample_emails]
    
    # Combine features
    combined_features = [combine_features(ret, tess) for ret, tess in zip(retvec_features, tesseract_predictions)]

    # Train Random Forest model
    classifier = RandomForestClassifier()
    classifier.fit(combined_features, sample_labels)
    joblib.dump(classifier, 'random_forest_model.pth')  
    print("Random Forest model trained and saved.")


# Helper functions
def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

def extract_text_from_file(file_path):
    email_text = ""
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # Extract email body
    for part in msg.walk():
        content_type = part.get_content_type()
        charset = part.get_content_charset() or 'utf-8'
        if content_type == 'text/plain':
            payload = part.get_payload(decode=True)
            if payload:
                email_text += payload.decode(charset, errors='ignore') + "\n"
        elif content_type == 'text/html':
            payload = part.get_payload(decode=True)
            if payload:
                html_content = payload.decode(charset, errors='ignore')
                soup = BeautifulSoup(html_content, 'html.parser')
                email_text += soup.get_text() + "\n"
        elif part.get_content_disposition() == 'attachment':
            filename = part.get_filename()
            if filename and filename.lower().endswith(('.txt', '.eml', '.docx')):
                attachment_path = os.path.join('attachments', filename)
                os.makedirs('attachments', exist_ok=True)
                with open(attachment_path, 'wb') as attachment_file:
                    attachment_file.write(part.get_payload(decode=True))
                extracted_text = extract_text_from_image(attachment_path)
                email_text += f"\n[Extracted Text from {filename}]:\n{extracted_text}\n"
                os.remove(attachment_path)
    return email_text

def get_retvec_features(email_text, model, tokenizer, max_length=128, target_size=1266):
    try:
        # Tokenize the email text
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

        # Model inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Ensure outputs contain the required embeddings
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            raise ValueError("Model outputs do not contain expected `logits` and `text_embedding`.")
        
        logits, text_embedding = outputs
        if text_embedding is None:
            raise ValueError("Text embedding is None.")
        
        embedding = text_embedding.squeeze().numpy()

        # Ensure embedding size matches target_size (pad or truncate)
        if embedding.shape[0] < target_size:
            padding = np.zeros(target_size - embedding.shape[0])  # Pad with zeros
            embedding = np.concatenate([embedding, padding])
        elif embedding.shape[0] > target_size:
            embedding = embedding[:target_size]  # Truncate if necessary

        return embedding
    except Exception as e:
        print(f"Error in get_retvec_features: {e}")
        raise

# Initialize the email data
email_data = {}

# Save email_data to a file
def save_email_data():
    with open('email_data.json', 'w') as f:
        json.dump(email_data, f)

# Load email_data from a file
def load_email_data():
    global email_data
    if os.path.exists('email_data.json'):
        with open('email_data.json', 'r') as f:
            email_data = json.load(f)

def apply_nudging(user_id, email_text, model_prediction):
    # Load existing email interaction data
    load_email_data()

    # Ensure the user exists in the data
    if user_id not in email_data:
        email_data[user_id] = {}

    # Increment interaction count for this email
    if email_text not in email_data[user_id]:
        email_data[user_id][email_text] = 0

    email_data[user_id][email_text] += 1

    # Save the updated data
    save_email_data()

    # Determine the nudging result
    if email_data[user_id][email_text] > 5:
        return 'Ham'  # User-specific behavior overrides
    else:
        return 'Spam' if model_prediction == 1 else 'Ham'
    
# front page route
@app.route('/')
def front_page():
    return render_template('front.html')

@app.route('/uploads/<filename>')
def serve_image(filename):
    # Path to the folder where the image is stored
    image_folder = os.path.join(app.root_path, 'uploads')
    return send_from_directory(image_folder, filename)
@app.route('/uploads/<filename>')
def serve_video(filename):
    video_folder = os.path.join(app.root_path, 'uploads')
    return send_from_directory(video_folder, filename)

@app.route('/login', methods=['GET'])
def show_login_page():
    return render_template('login.html')

@app.route('/demo', methods=['GET'])
def show_demo_page():
    return render_template('demo.html')

# Sign Up Route
@app.route('/signup', methods=['POST'])
def signup():
    try:
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')

        if not email or not name or not password:
            return jsonify({"error": "All fields are required."}), 400

         # Check if user already exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT password FROM user_cl WHERE email = %s", (email,))
        existing_user = cur.fetchone()
        if existing_user:
            # Optional: Check if the provided password matches the hash of the existing user
            if bcrypt.checkpw(password.encode('utf-8'), existing_user[0].encode('utf-8')):
                return jsonify({"error": "User with this email already exists and password matches."}), 400
            else:
                return jsonify({"error": "User with this email already exists but the password does not match."}), 400
            
        # Hash the provided password    
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())    

        # Insert new user
        cur.execute("INSERT INTO user_cl (email, name, password) VALUES (%s, %s, %s)", (email, name, password_hash))
        mysql.connection.commit()
        cur.close()

        # Return success response with API key
        api_key = "lXnaPparqHTKjpDhRg-DVjaQv1Ep-cHa-YjodckPqS0"
        session['name'] = name
        session['email'] = email
        return jsonify({"message": "User registered successfully!", "username": name, "apiKey": api_key, "redirect": "/dashboard"}), 201

    except Exception as e:
        print(f"Error during signup: {e}")
        return jsonify({"error": "An error occurred during signup."}), 500

# Sign In Route
@app.route('/signin', methods=['POST'])
def signin():
    try:
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return jsonify({"error": "Email and password are required."}), 400

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name, email, password FROM user_cl WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session['id'] = user[0]
            session['name'] = user[1]
            session['email'] = user[2]

            api_key = "lXnaPparqHTKjpDhRg-DVjaQv1Ep-cHa-YjodckPqS0"
            return jsonify({
                "message": "Login successful!",
                "username": user[1],
                "redirect": "/dashboard",
                "apiKey": api_key
            }), 200
        else:
            return jsonify({"error": "Invalid email or password."}), 401

    except Exception as e:
        print(f"Error during signin: {e}")
        return jsonify({"error": "An error occurred during signin."}), 500

    
# Function to check if a file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # Ensure user is logged in by checking session
    if 'name' not in session:
        return redirect(url_for('show_login_page'))
    
    final_result = None  # Initialize result variable
    spam_count = 0
    ham_count = 0
    
    # Handle POST request to upload and classify an email
    if request.method == 'POST':
        # Get the Authorization header
        api_key = request.headers.get('Authorization')
        if not api_key:
            return jsonify({"error": "Authorization header missing"}), 401

        try:
            # Extract API key from Authorization header
            api_key = api_key.split(" ")[1]
        except IndexError:
            return jsonify({"error": "Invalid Authorization header format"}), 401

        # Check if the API key is correct
        if api_key != "lXnaPparqHTKjpDhRg-DVjaQv1Ep-cHa-YjodckPqS0":
            return jsonify({"error": "Unauthorized access"}), 403
    
        # Handle file processing and classification (same logic as before)
        file = request.files.get('email_file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type or no file provided'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            # Extract text from the email file
            email_text = extract_text_from_file(file_path)

            # Extract features from RETVec model
            retvec_features = get_retvec_features(email_text, retvec_model, tokenizer)

            # Extract prediction from Tesseract model
            tesseract_prediction = classify_with_tesseract(email_text)
            # Convert Tesseract prediction to numeric (1 for Spam, 0 for Ham)
            tesseract_features = [1 if tesseract_prediction == 'Spam' else 0]

            # Combine RETVec and Tesseract features into a 1269-dimensional feature vector
            combined_features = combine_features(retvec_features, tesseract_features)

            # Predict using Random Forest classifier
            prediction = classifier.predict(combined_features.reshape(1, -1))

            # Apply nudging if necessary
            final_result = apply_nudging(session.get('email', ''), email_text, prediction[0])

            # Determine counts for spam and ham
            if final_result == 'Spam':
                spam_count = 1
            else:
                ham_count = 1
            
            # Save classification in the database
            cur = mysql.connection.cursor()
            cur.execute("SELECT COUNT(*) FROM spam_history WHERE email = %s", (session['email'],))
            exists = cur.fetchone()[0]

            if exists == 0:
                cur.execute(
                    "INSERT INTO spam_history(email_text, label, email) VALUES (%s, %s, %s)",
                    (email_text, final_result, session['email'])
                )
            else:
                cur.execute(
                    "UPDATE spam_history SET email_text = %s, label = %s WHERE email = %s",
                    (email_text, final_result, session['email'])
                )

            mysql.connection.commit()
            cur.close()

            return jsonify({
                'result': final_result,
                'spamCount': spam_count,
                'hamCount': ham_count
            })    
        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({'error': 'Error processing file'}), 500

    return render_template('dashboard.html', name=session['name'], final_result=final_result, spam_count=spam_count, ham_count=ham_count)
    
@app.route('/spam_history', methods=['GET', 'POST'])
def spam_history():
    # Check if the request is POST (for updating classification)
    if request.method == 'POST':
        try:
            data = request.get_json()
            email_id = data['id']  # Use `id` from the request payload
            new_label = data['label']  # Use `label` from the request payload

            # Update the label in the database
            cur = mysql.connection.cursor()
            cur.execute("UPDATE spam_history SET label = %s WHERE id = %s", (new_label, email_id))
            mysql.connection.commit()

            # Retrieve updated counts for spam and inbox
            user_email = session['email']
            cur.execute("SELECT COUNT(*) FROM spam_history WHERE email = %s AND label = 'ham'", (user_email,))
            inbox_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM spam_history WHERE email = %s AND label = 'spam'", (user_email,))
            spam_count = cur.fetchone()[0]

            # Optionally retrieve the email text for reclassification (if needed)
            cur.execute("SELECT email_text FROM spam_history WHERE id = %s", (email_id,))
            email_text = cur.fetchone()[0]
            cur.close()

            #Append reclassified data to the training CSV file
            training_file_path = 'C:\\Users\\samri\\Downloads\\email_classification.csv'
            if os.path.exists(training_file_path):
                training_data = pd.read_csv(training_file_path)
            else:
                training_data = pd.DataFrame(columns=['email', 'label'])

            new_data = pd.DataFrame({'email': [email_text], 'label': [new_label]})
            training_data = pd.concat([training_data, new_data], ignore_index=True)
            training_data.to_csv(training_file_path, index=False)

            return jsonify({
                'status': 'success',
                'message': 'Email reclassified successfully and model updated!',
                'inbox_count': inbox_count,
                'spam_count': spam_count
            }), 200

        except Exception as e:
            print(f"Error reclassifying email: {e}")
            return jsonify({'status': 'error', 'message': 'Error reclassifying email'}), 500

    # Handle GET requests to fetch spam history and counts
    else:
        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT email_text, label, id FROM spam_history WHERE email = %s", (session['email'],))
            spam_history = cur.fetchall()

            cur.execute("SELECT COUNT(*) FROM spam_history WHERE email = %s AND label = 'ham'", (session['email'],))
            inbox_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM spam_history WHERE email = %s AND label = 'spam'", (session['email'],))
            spam_count = cur.fetchone()[0]
            cur.close()

            # Render the spam history page with counts and email history
            return render_template('spam_history.html', username=session['name'], spam_history=spam_history,
                                   inbox_count=inbox_count, spam_count=spam_count)
        except Exception as e:
            print(f"Error fetching spam history: {e}")
            return jsonify({'status': 'error', 'message': 'Error fetching spam history'}), 500

@app.route('/fetch_emails/<category>', methods=['GET'])
def fetch_emails(category):
    if 'email' not in session:
        return jsonify({'error': 'User not logged in'}), 403

    user_email = session['email']
    try:
        cur = mysql.connection.cursor()
        if category == 'inbox':
            cur.execute("SELECT id, email_text, label FROM spam_history WHERE email = %s AND label = 'ham'", (user_email,))
        elif category == 'spam':
            cur.execute("SELECT id, email_text, label FROM spam_history WHERE email = %s AND label = 'spam'", (user_email,))
        else:
            return jsonify({'error': 'Invalid category'}), 400

        emails = cur.fetchall()
        email_list = [{'id': email[0], 'body': email[1], 'label': email[2]} for email in emails]

        # Fetch updated inbox and spam counts
        cur.execute("SELECT COUNT(*) FROM spam_history WHERE email = %s AND label = 'ham'", (user_email,))
        inbox_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM spam_history WHERE email = %s AND label = 'spam'", (user_email,))
        spam_count = cur.fetchone()[0]
        cur.close()

        # Return the email list and counts as JSON
        return jsonify({
            'emails': email_list,
            'inbox_count': inbox_count,
            'spam_count': spam_count
        }), 200
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return jsonify({'error': 'Error fetching emails'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
