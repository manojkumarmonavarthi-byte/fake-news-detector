import sqlite3

conn = sqlite3.connect("users.db", check_same_thread=False)

c = conn.cursor()

def create_usertable():
    c.execute(
        "CREATE TABLE IF NOT EXISTS users(username TEXT,password TEXT)"
    )

def add_userdata(username, password):
    c.execute(
        "INSERT INTO users(username,password) VALUES (?,?)",
        (username, password),
    )
    conn.commit()

def login_user(username, password):
    c.execute(
        "SELECT * FROM users WHERE username =? AND password = ?",
        (username, password),
    )

    data = c.fetchall()

    return data

def view_all_users():
    c.execute("SELECT * FROM users")

    data = c.fetchall()

    return data