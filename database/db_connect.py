import sqlite3 as sql
from werkzeug.security import generate_password_hash, check_password_hash


__all__ = ['get_registered_users', 'create_db', 'drop_db', 'add_user_and_passw', 'check_user_and_passw', 'get_user_and_passw', 'get_user_id']





def create_db():
    try:
        conn = sql.connect('database.db')
        conn.execute('''CREATE TABLE IF NOT EXISTS tb_user 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        username TEXT UNIQUE NOT NULL, 
                        password TEXT NOT NULL,
                        training_count INTEGER DEFAULT 0)''')
        print("INFO - DATABASE - Table created successfully or already exists")
        conn.close()
    except Exception as e:
        print(f"ERROR - DATABASE - {str(e)}")

# Add this new function to add the training_count column to existing tables
def add_training_count_column():
    try:
        conn = sql.connect('database.db')
        conn.execute('''ALTER TABLE tb_user ADD COLUMN training_count INTEGER DEFAULT 0''')
        print("INFO - DATABASE - Column 'training_count' added successfully")
        conn.close()
    except Exception as e:
        if 'duplicate column name' in str(e):
            print("INFO - DATABASE - Column 'training_count' already exists")
        else:
            print(f"ERROR - DATABASE - {str(e)}")
def user_exists(username):
    try:
        con = sql.connect("database.db")
        cur = con.cursor()
        cur.execute("SELECT * FROM tb_user WHERE username=?", (username,))
        user = cur.fetchone()
        con.close()
        return user is not None
    except Exception as e:
        print(f"Error checking if user exists: {str(e)}")
        return False

def drop_db():
    try:
        conn = sql.connect('database.db')
        conn.execute('DROP TABLE tb_user')
        print("INFO - DATABASE - Table deleted successfully!")
        conn.close()
    except:
        print("ERROR - DATABASE - Connection denied or database does not exist")

def add_user_and_passw(username, password):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM tb_user WHERE username=?", (username,))
            if cur.fetchone():
                print(f"Username '{username}' already exists!")
                return 0, False
            else:
                hashed_password = generate_password_hash(password)
                cur.execute("INSERT INTO tb_user (username, password) VALUES (?, ?)", (username, hashed_password))
                con.commit()
                id = cur.lastrowid
                print(f"User created, ID: {id}")
                return id, True
    except Exception as e:
        print(f"Error in registration: {str(e)}")
        return 0, False

def check_user_and_passw(username, password):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("SELECT id, password FROM tb_user WHERE username=?", (username,))
            user = cur.fetchone()
            if user and check_password_hash(user[1], password):
                print(f"WARNING - DATABASE - check_user_and_pass: Username and password match, user authenticated! USER_ID: {user[0]}, USER_NAME: {username}")
                return 1, True, user[0]
            else:
                print("WARNING - DATABASE - check_user_and_pass: Username or password incorrect!")
                return 1, False, None
    except Exception as e:
        print(f"Error in authentication: {str(e)}")
        return 3, False, None

def get_user_and_passw(id):
    try:
        con = sql.connect("database.db")
        con.row_factory = sql.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM tb_user")
   
        rows = cur.fetchall()
        username = rows[id]['username']
        password = rows[id]['password']
        return username, password
    except:
        print("Unable to retrieve users!")

def get_user_id(username):
    try:
        con = sql.connect("database.db")
        con.row_factory = sql.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM tb_user WHERE username=?", (username,))
    
        rows = cur.fetchall()
        if rows:
            for row in rows:
                user_id = row[0]
        return user_id
    except:
        print("Unable to retrieve USER_ID from the database!")

def get_registered_users():
    try:
        conn = sql.connect('database.db')
        cur = conn.cursor()
        cur.execute("SELECT id, username FROM tb_user")
        users = cur.fetchall()
        conn.close()
        return users
    except Exception as e:
        print(f"ERROR - DATABASE - Unable to retrieve registered users: {str(e)}")
        return []



'''
def is_user_trained(username):
    try:
        conn = sql.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT is_trained FROM tb_user WHERE username=?", (username,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else False
    except Exception as e:
        print(f"ERROR - DATABASE - Unable to check user training status: {str(e)}")
        return False

def update_user_training_status(username):
    try:
        conn = sql.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE tb_user SET is_trained = TRUE WHERE username=?", (username,))
        conn.commit()
        conn.close()
        print(f"INFO - DATABASE - Training status updated for user: {username}")
    except Exception as e:
        print(f"ERROR - DATABASE - Unable to update user training status: {str(e)}")

def add_is_trained_column():
    try:
        conn = sql.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE tb_user ADD COLUMN is_trained BOOLEAN DEFAULT FALSE")
        conn.commit()
        conn.close()
        print("INFO - DATABASE - Column 'is_trained' added successfully")
    except Exception as e:
        if 'duplicate column name' in str(e):
            print("INFO - DATABASE - Column 'is_trained' already exists")
        else:
            print(f"ERROR - DATABASE - {str(e)}")

def update_training_status(user_id):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("UPDATE tb_user SET is_trained = TRUE WHERE id = ?", (user_id,))
            con.commit()
            print(f"User {user_id} training status updated")
    except Exception as e:
        print(f"Error updating training status: {str(e)}")
'''
def update_training_count(username):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("UPDATE tb_user SET training_count = COALESCE(training_count, 0) + 1 WHERE username = ?", (username,))
            con.commit()
    except Exception as e:
        print(f"Error updating training count: {str(e)}")

def get_training_count(username):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("SELECT COALESCE(training_count, 0) FROM tb_user WHERE username = ?", (username,))
            result = cur.fetchone()
            return result[0] if result else 0
    except Exception as e:
        print(f"Error getting training count: {str(e)}")
        return 0
if __name__ == '__main__':
    create_db()
