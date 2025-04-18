# Spam-Email-DEtection
📧 Spam Email Detection using Machine Learning
------------------------------------------------
This project is a machine learning-based system to detect and classify spam emails. By analyzing the contents of emails, the model can determine whether a message is spam or legitimate (ham). This is crucial for reducing phishing attacks, scams, and unwanted email clutter.

🚀 Features
------------------
Classifies emails into Spam or Ham.

Utilizes text preprocessing and vectorization (e.g., TF-IDF, RETVec).

Deep learning model built using PyTorch.

Web interface using Streamlit for easy interaction.

Email upload support (.txt, .eml formats).

Stores classified emails in a MySQL database.

Secure user authentication with hashed passwords.

🧠 Model
-----------
The model is trained on a labeled dataset of emails.

Uses a RETVec-based tokenizer for better handling of adversarial spam.

Trained using binary classification (Spam = 1, Ham = 0).

🛠️ Tech Stack
--------------------
Frontend: Streamlit

Backend: Flask (for DB interactions)

Database: MySQL

ML Framework: PyTorch

Tokenizer: RETVec

Authentication: bcrypt (for password hashing)

🧪 Usage
------------
Sign up / Login

Upload a .txt or .eml email file.

Get prediction: Spam or Ham.

Track past predictions in the transaction history section.

View analytics with transaction graphs.

🔒 Security
--------------
Passwords are securely hashed using bcrypt.

Input validation and sanitization are applied during file upload and DB interaction.

📈 Future Improvements
---------------------------
Add support for PDF/email image attachments using OCR.

Integrate with email clients (e.g., Gmail API).

Improve accuracy with ensemble models.

Add user roles and admin dashboard.
