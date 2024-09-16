from database.db_connect import get_registered_users
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import joblib
import sqlite3
import os

def get_registered_users():
    registered_users = []
    try:
        db_path = r'C:\Users\USER\.vscode\keystrokeDynamics2FA\webservice\database\users.db'
        
        if not os.path.exists(db_path):
            print(f"Database file not found at {db_path}")
            return registered_users

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users")
        registered_users = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"Error reading registered users: {str(e)}")
    return registered_users


import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler





class KNNClassifier:
    def __init__(self, data_path, typing_sample, split, k):
        self.registered_biometric_file = data_path
        self.typing_sample = typing_sample
        self.split = split
        self.neighbour_size = k
        self.X = []
        self.y = []
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=self.neighbour_size, metric="manhattan")
        self.feature_names = []
        self.training_count = 0
        self.RETRAIN_THRESHOLD = 10  # Adjust this value as needed
        self.load_data_from_csv()


    def load_data_from_csv(self):
        keystroke_data = pd.read_csv(self.registered_biometric_file)
        self.X = keystroke_data.iloc[:, :85].values.astype(float)  # First 85 columns are features
        self.y = keystroke_data.iloc[:, 85].values  # Last column is the username

    def predict(self, new_data):
        if len(self.X) < 2:  # We need at least 2 samples to make a prediction
            return None, 0  # Not enough training data

        new_data = np.array(new_data).reshape(1, -1)
        if new_data.shape[1] != self.X.shape[1]:
            print(f"Error: Input data has {new_data.shape[1]} features, but model expects {self.X.shape[1]}")
            return None, 0

        try:
            new_data_scaled = self.scaler.transform(new_data)
            prediction = self.knn.predict(new_data_scaled)
            probabilities = self.knn.predict_proba(new_data_scaled)
            max_probability = np.max(probabilities)
            return prediction[0], max_probability
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, 0

    def knn_manhattan_holdout(self): # metric is given by holdout
        description = 'knn_manhattan_t'
        keystroke_data = pd.read_csv(self.registered_biometric_file, keep_default_na=False)
        
        sample = self.typing_sample

        # Should be changed when the array quantity is different from 121
        data = keystroke_data.iloc[:, 0:85]  

        # Classes for application in supervised learning
        target = keystroke_data['username']

        sample_dataframe = pd.DataFrame.transpose(pd.DataFrame(sample))

        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=self.knn_model_test_ratio, random_state=10)

        knn_model = KNeighborsClassifier(n_neighbors=self.neighbour_size, metric="manhattan")

        knn_model.fit(data_train, target_train)

        inner_prediction = knn_model.predict(data_test)
        user_predict = knn_model.predict(sample_dataframe)

        accuracy = accuracy_score(target_test, inner_prediction)

        print("[+] Predicted user:", user_predict)
        
        return str(user_predict), str(accuracy), description

    def knn_manhattan_without_training(self):
        print("Starting KNN Manhattan classification...")
        try:
            # Load the data
            data = pd.read_csv(self.registered_biometric_file)
            print(f"Data shape: {data.shape}")
            print(f"Data columns: {data.columns}")

            # Check if 'username' column exists
            if 'username' not in data.columns:
                print("'username' column not found. Available columns:", data.columns)
                return None

            description = 'knn_manhattan_s'
            keystroke_data = data
            print("Keystroke data shape:", keystroke_data.shape)
            sample = self.typing_sample

            # Should be changed when the array quantity is different from 121
            data = keystroke_data.iloc[:, 0:85]
            print("Data shape after slicing:", data.shape)
            if len(self.typing_sample) != 85:
                print(f"Warning: Expected 85 features, got {len(self.typing_sample)}")
                self.typing_sample = self.typing_sample[:85] + [0] * (85 - len(self.typing_sample))
            print("Typing sample length after adjustment:", len(self.typing_sample))
            # Update feature names if they differ from the CSV
            self.feature_names = data.columns.tolist()
            # Classes for application in supervised learning
            target = keystroke_data['username']
            # Ensure only registered users are in the training set
            registered_users = get_registered_users()
            if not registered_users:
                print("No registered users found. Using all data.")
            else:
                data = data[target.isin(registered_users)]
                target = target[target.isin(registered_users)]
            
            if data.empty:
                print("No data available after filtering. Using all data.")
                data = keystroke_data.iloc[:, 0:85]
                target = keystroke_data['username']
            
            # Convert empty strings to NaN and then to 0
            data = data.replace('', np.nan).fillna(0)
            # Convert all data to float
            data = data.astype(float)
            print("Final data shape:", data.shape)
            print("Target shape:", target.shape)
            # Create DataFrame for the typing sample with feature names
            sample_df = pd.DataFrame([self.typing_sample], columns=self.feature_names)
            print("Sample DataFrame shape:", sample_df.shape)
            
            if len(data) < 2:
                print("Not enough data for classification. At least 2 samples are required.")
                return None
            
            knn_model = KNeighborsClassifier(n_neighbors=min(self.neighbour_size, len(data)), metric="manhattan")

            knn_model.fit(data, target)

            inner_prediction = knn_model.predict(sample_df)
            accuracy = accuracy_score(target, [inner_prediction[0]] * len(target))

            print(f"Classification result: {inner_prediction[0]}")
            print('[+] Predicted User - ', inner_prediction[0])
            
            # After fitting the model
            self.training_count += 1
            self.update_model()
     
            return str(inner_prediction[0]), str(accuracy), description

        except Exception as e:
            print(f"Error in knn_manhattan_without_training: {str(e)}")
            return None
            return None

    def hyper_parameters_tuning(self):
        keystroke_data = pd.read_csv(self.registered_biometric_file, keep_default_na=False)
        
        # Should be changed when the array quantity is different from 121
        data = keystroke_data.iloc[:, 0:85]  

        # Classes for application in supervised learning
        target = keystroke_data['username']

        k_range = list(range(1, 10))
        leaf_size = list(range(1, 50))
        weight_options = ["uniform", "distance"]
        p = [1, 2]

        param_grid = dict(leaf_size=leaf_size, n_neighbors=k_range, weights=weight_options, p=p)

        knn = KNeighborsClassifier()

        grid = GridSearchCV(knn, param_grid, scoring='accuracy')
        grid.fit(data, target)

        best_score = grid.best_score_
        best_params = grid.best_params_
        best_estimator = grid.best_estimator_

        print('[+] Best Score - ', best_score)
        print('[+] Best Params - ', best_params)
        print('[+] Best Estimator - ', best_estimator)
 
        return best_score, best_params, best_estimator

    def get_cv_score(self):
        description = 'knn_manhattan_score_test'
        keystroke_data = pd.read_csv(self.registered_biometric_file, keep_default_na=False)

        # Should be changed when the array quantity is different from 121
        data = keystroke_data.iloc[:, 0:85]  # Change this to 85

        # Convert empty strings to NaN and then to 0
        data = data.replace('', np.nan).fillna(0)
        
        # Convert all data to float
        data = data.astype(float)

        # Classes for application in supervised learning
        target = keystroke_data['username']

        knn_model = KNeighborsClassifier(n_neighbors=self.neighbour_size, metric="manhattan")

        knn_model.fit(data, target)

        scores = cross_val_score(knn_model, data, target, scoring=['accuracy'])
        score_result = scores['test_accuracy'].mean() * 100
        print('[+] Average Accuracy (test_accuracy): %.2f' % score_result)
 
        return score_result

    def predict_user(self, sample):
        if not sample or len(sample) == 0:
            print("Error: Empty sample received for prediction")
            return "Unknown User"

        sample_df = pd.DataFrame([sample], columns=self.feature_names)
        if sample_df.empty:
            print("Error: Failed to create DataFrame from sample")
            return "Unknown User"

        try:
            distances, indices = self.knn_model.kneighbors(sample_df)
            nearest_classes = self.target[indices[0]]
            
            # Get the most common class and its frequency
            predicted_class = max(set(nearest_classes), key=list(nearest_classes).count)
            confidence = list(nearest_classes).count(predicted_class) / len(nearest_classes)
            
            if confidence < 0.6:  # Adjust this threshold as needed
                return "Unknown User"
            else:
                return predicted_class
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Unknown User"

    def update_model(self):
        if self.training_count > self.RETRAIN_THRESHOLD:
            self.train_model()
            self.training_count = 0

    def train_model(self):
        # Read the latest data from the biometrics CSV file
        latest_data = pd.read_csv(self.registered_biometric_file, header=0)

        
        if latest_data.empty:
            print("No training data available in the CSV file.")
            return
            
        self.training_count = 0
        self.X = latest_data.iloc[:, 0:85].values
        self.y = latest_data['username'].values
        self.feature_names = latest_data.columns[:85].tolist()
    
        self.X = self.scaler.fit_transform(self.X)
        self.knn.fit(self.X, self.y)
        
        # Assume the last column is the username/user_id
        X = latest_data.iloc[:, :-1]
        y = latest_data.iloc[:, -1]
        
        # Update feature names
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Create and train a new KNN model
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k, metric='manhattan')
        self.knn_model.fit(X, y)
        
        # Update the target values
        self.target = y.values
        
        print("Model successfully retrained with latest keyboard input data.")
        
        # Reset the training count
        

    def validate_training(self, username, new_sample):
        prediction = self.predict_user(new_sample)
        if prediction != username:
            return False
        return True

    def update_model(self):
        if self.training_count > self.RETRAIN_THRESHOLD:
         self.train_model()

    #replaced all instances of "CLASS" with 'username"