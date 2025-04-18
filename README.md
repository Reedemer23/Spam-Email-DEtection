# Spam-Email-DEtection
📧 Spam Email Detection using Machine Learning
------------------------------------------------
This project is a machine learning-based system to detect and classify spam emails. By analyzing the contents of emails, the model can determine whether a message is spam or legitimate (ham). This is crucial for reducing phishing attacks, scams, and unwanted email clutter.

🚀 Features
------------------
1)Classifies emails into Spam or Ham.
2)Utilizes text preprocessing and vectorization (e.g., TF-IDF, RETVec).
3)Deep learning model built using PyTorch.
4)Web interface using Streamlit for easy interaction.
5)Email upload support (.txt, .eml formats).
6)Stores classified emails in a MySQL database.
7)Secure user authentication with hashed passwords.

🧠 Model
-----------
1)The model is trained on a labeled dataset of emails.
2)Uses a RETVec-based tokenizer for better handling of adversarial spam.
3)Trained using binary classification (Spam = 1, Ham = 0).

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
1)Sign up / Login
2)Upload a .txt or .eml email file.
3)Get prediction: Spam or Ham.
4)Track past predictions in the transaction history section.
5)View analytics with transaction graphs.

🔒 Security
--------------
1)Passwords are securely hashed using bcrypt.
2)Input validation and sanitization are applied during file upload and DB interaction.

📈 Future Improvements
---------------------------
1)Add support for PDF/email image attachments using OCR.
2)Integrate with email clients (e.g., Gmail API).
3)Improve accuracy with ensemble models.
4)Add user roles and admin dashboard.
