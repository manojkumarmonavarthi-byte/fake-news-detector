import sqlite3
import hashlib

conn = sqlite3.connect(
    "users.db",
    check_same_thread=False
)

c = conn.cursor()

def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()

def create_tables():

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        news TEXT,
        result TEXT
    )
    """)

    conn.commit()

def add_user(username, password):

    try:

        c.execute(
            "INSERT INTO users(username,password) VALUES(?,?)",
            (
                username,
                hash_password(password)
            )
        )

        conn.commit()

        return True

    except:
        return False

def login_user(username, password):

    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (
            username,
            hash_password(password)
        )
    )

    return c.fetchone()

def add_history(username, news, result):

    c.execute(
        "INSERT INTO history(username,news,result) VALUES(?,?,?)",
        (
            username,
            news,
            result
        )
    )

    conn.commit()

def get_history(username):

    c.execute(
        "SELECT news,result FROM history WHERE username=? ORDER BY id DESC",
        (username,)
    )

    return c.fetchall()