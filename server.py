# IMPORTS OF CUSTOM LIBRARIES

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from csv_fixer import fix_csv
import csv_fixer

from database.db_connect import (
    get_user_id, drop_db, create_db, add_user_and_passw, check_user_and_passw,
    update_training_count, get_training_count, add_training_count_column, user_exists  # Add this import
)
from knn_sdk.KNNClassifier import KNNClassifier
import datetime
import csv
import sqlite3 as sql
from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sqlite3

# Define the base directory as the webservice folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for database and biometrics file
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')
BIOMETRICS_FILE = os.path.join(BASE_DIR, 'database', 'biometrics.csv')

def initialize_database():
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# Ensure the directory for BIOMETRICS_FILE exists
    if not os.path.exists(os.path.dirname(BIOMETRICS_FILE)):
        os.makedirs(os.path.dirname(BIOMETRICS_FILE))

# Call this function when your app starts
initialize_database()

# ... (previous code remains unchanged)

initialize_database()

def populate_existing_users():
    # This is just an example. Replace with your actual user data.
    users = [
        ("oluwatomiwa", "oluwatomiwa"),
        ("kkk", "kkk"),
        # Add more users as needed
    ]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for username, password in users:
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            print(f"Added user: {username}")
        except sqlite3.IntegrityError:
            print(f"User {username} already exists")
    conn.commit()
    conn.close()

# Uncomment the next line to populate the database with existing users
# populate_existing_users()

BIOMETRICS_FILE = r'C:\Users\USER\.vscode\keystrokeDynamics2FA\webservice\database\biometrics.csv'
LOG_NAME = 'results.log'
K = 1
SPLIT = 0.8
app = Flask(__name__, template_folder='templates', static_folder='./static')
create_db()
add_training_count_column()  # Add this line



def check_db_connection():
    try:
        conn = sql.connect('database.db')
        conn.close()
        print("Database connection successful")
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        sys.exit(1)

check_db_connection()

@app.route('/')
def home():
    return render_template('./home/home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup/signup.html')
    elif request.method == 'POST':
        print("Received POST request")
        try:
            response = dict(request.get_json())
            print("Response:", response)
            username = response['username']
            password = response['password']
            print(f"Username: {username}, Password: {password}")
            id, result = add_user_and_passw(username, password)
            print(f"ID: {id}, Result: {result}")

            if result:
                return jsonify({'signup_code': 'UserRegistrySuccess', 'user_id': id, 'message': 'User registered successfully!'})
            else:
                return jsonify({'signup_code': 'UsernameAlreadyExist', 'message': 'Username already exists or an error occurred. Please check the server logs for details.'})
        except Exception as e:
            print(f"Error in signup route: {str(e)}")
            return jsonify({'error': str(e), 'message': 'An error occurred during registration.'}), 500

@app.route('/signup/biometrics', methods=['POST'])
def signup_biometrics():
    if request.method == 'POST':
        print("Received biometric data POST request")
        response = dict(request.get_json())
        user_id = response['user_id']
        data = response['data']
        data.append(user_id)  # add user id to the end of the list
        print(f"User ID: {user_id}")
        print(f"Data to be saved: {data}")
        try:
            with open(BIOMETRICS_FILE, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            print(f"Data successfully written to {BIOMETRICS_FILE}")
            return jsonify({'biometric_code': 'Success'})
        except Exception as e:
            print(f"Error writing to file: {str(e)}")
            return jsonify({'biometric_code': 'Unable to register biometric data', 'error': str(e)})

@app.route('/authenticate', methods=['GET'])
def authenticate():
    return render_template('./authenticate/authenticate.html')

@app.route('/authenticate/auth1', methods=['POST'])  # Route for the first authentication
def auth1():
    response = dict(request.get_json())
    username = response['username']
    password = response['password']

    id, result, user_id = check_user_and_passw(username, password)

    if result:
        return jsonify({'auth1_code': 'success', 'user_id': user_id})
    else:
        if id == 3:
            return jsonify({'auth1_code': 'UsernameNotExist'})
        elif id == 1:
            return jsonify({'auth1_code': 'PasswordIsWrong'})

@app.route('/authenticate/auth2', methods=['POST'])
def auth2():
    try:
        data = request.get_json()
        username = data['username']
        biometric_data = data['data']

        # Ensure biometric_data has 85 features
        if len(biometric_data) < 85:
            biometric_data.extend([0] * (85 - len(biometric_data)))
        elif len(biometric_data) > 85:
            biometric_data = biometric_data[:85]

        typing_sample = [float(x) if x != '' else 0.0 for x in biometric_data]

        print("Username:", username)
        print("Typing sample length:", len(typing_sample))

        # Load the training data
        if not os.path.exists(BIOMETRICS_FILE):
            raise FileNotFoundError(f"Biometrics file not found: {BIOMETRICS_FILE}")

        df = pd.read_csv(BIOMETRICS_FILE, header=0)
        print("Training data shape:", df.shape)

        knn_classifier = KNNClassifier(BIOMETRICS_FILE, typing_sample, SPLIT, K)
        
        # Add these debug print statements here
        print(f"BIOMETRICS_FILE path: {BIOMETRICS_FILE}")
        print(f"File exists: {os.path.exists(BIOMETRICS_FILE)}")
        
        result = knn_classifier.knn_manhattan_without_training()

        if result is None:
            return jsonify({'error': 'Classification failed. Check server logs for details.'}), 400

        predicted_user, accuracy, description = result

        match = (username == predicted_user)

        # Log the authentication attempt
        current_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_entry = f"[+] Real User: {username} | Predicted User: {predicted_user} | "
        log_entry += f"Algorithm: {description} | K Value: {K} | Match: {match} | "
        log_entry += f"Accuracy: {accuracy} | Date: {current_datetime}\n"
        
        with open(LOG_NAME, 'a') as file:
            file.write(log_entry)

        response = {
            'username': username,
            'predicted_user': predicted_user,
            'accuracy': accuracy,
            'result': str(match),
            'algorithm': description
        }

        return jsonify(response), 200

    except Exception as e:
        error_message = f"Error in auth2: {str(e)}"
        print(error_message)
        current_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(LOG_NAME, 'a') as file:
            file.write(f"[!] Error: {error_message} | Date: {current_datetime}\n")
        return jsonify({'error': error_message}), 500

@app.route('/train/biometrics', methods=['POST'])
def train_biometrics():
    try:
        data = request.get_json()
        username = data['username']
        
        if not user_exists(username):
            return jsonify({'error': 'User not registered. Please sign up first.'}), 400
        
        biometric_data = data['data']
        
        print(f"Received training data for user: {username}")
        print(f"Biometric data length: {len(biometric_data)}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(BIOMETRICS_FILE), exist_ok=True)
        
        # Append the new training data to biometrics.csv
        file_exists = os.path.isfile(BIOMETRICS_FILE)
        
        with open(BIOMETRICS_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write header if file is new
                writer.writerow([f'feature_{i}' for i in range(1, 86)] + ['CLASS'])
            # Ensure biometric_data has exactly 85 features
            if len(biometric_data) < 85:
                biometric_data.extend([0] * (85 - len(biometric_data)))
            elif len(biometric_data) > 85:
                biometric_data = biometric_data[:85]
            writer.writerow(biometric_data + [username])  # Add username at the end
        
        update_training_count(username)
        count = get_training_count(username)
        
        print(f"Training data saved to {BIOMETRICS_FILE}")
        
        return jsonify({
            'message': 'Training data received and saved successfully',
            'training_count': count
        }), 200
    except Exception as e:
        print(f"Error in train_biometrics: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/best_params', methods=['GET'])
def best_params():
    return render_template('./best_params/best_params.html')

@app.route('/best_params/result', methods=['GET'])
def best_params_result():
    typing_sample = ''  # Artifice to allow the use of the class
    classifier = KNNClassifier(BIOMETRICS_FILE, typing_sample, 0.7, 3)
    best_score, best_params, best_estimator = classifier.hyper_parameters_tuning()

    current_datetime = datetime.datetime.now()
    current_date = current_datetime.strftime("%d/%m/%Y %H:%M:%S ")

    with open(LOG_NAME, 'a') as file:  # Create log file
        file.write('[+]  Best Score: ')
        file.write(str(best_score))
        file.write(' |  Best Params: ')
        file.write(str(best_params))
        file.write(' |  Best Estimator: ')
        file.write(str(best_estimator))
        file.write(' | Date: ')
        file.write(current_date)
        file.write('\n')

    return jsonify({'best_score': str(best_score), 'best_params': str(best_params), 'best_estimator': str(best_estimator)})

@app.route('/login', methods=['GET'])
def login():
    return render_template('./authenticate/authenticate.html')

def ensure_correct_csv():
    input_file = BIOMETRICS_FILE
    output_file = os.path.join(os.path.dirname(BIOMETRICS_FILE), 'biometrics_corrected.csv')
    fix_csv(input_file, output_file)
    os.replace(output_file, input_file)
    print(f"CSV file has been checked and corrected if necessary: {BIOMETRICS_FILE}")

@app.route('/train')
def train():
    return render_template('./training/training.html')

# Server Start
if __name__ == '__main__':
    ensure_correct_csv()
    initialize_database()
    # Uncomment the next line if you need to populate existing users
    # populate_existing_users()
    app.run(host='127.0.0.1', debug=True, port=3000)
    initialize_database()

def query_user_from_database(username):
    user_id = get_user_id(username)
    return user_id is not None

def train_user_biometrics(username, biometric_data):
    # Your existing training code
    # ...
    
    # Log the training event
    log_training_event(username)
    
    # Update user's training count in database
    update_user_training_count(username)

def log_training_event(username):
    # Implement logging logic here
    pass

def update_user_training_count(username):
    # Implement database update logic here
    pass

# Add this after the app initialization
# this was part of the initial codebase


# app.run(host='127.0.0.1', debug=True, port=3000) was modified to app.run() in line 333
   
# i added run_with_ngrok(app) in line 89
# from flask_ngrok2 import run_with_ngrok at the op of the file
# i added the import flask from Flask line too 
# i added os.environ["NGROK_AUTH_TOKEN"] = "2m4ZFuSO6kew9Vm2ekxJDb95jxF_4BCEtMqHYbD5dA5QWV7Mu" to line 94

