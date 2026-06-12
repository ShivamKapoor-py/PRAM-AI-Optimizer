import sqlite3
import uuid
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. ALLOW CORS (So the browser can talk to the server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. MATCH THE ROUTE EXACTLY
@app.get("/v1/register") # Ensure there are no trailing slashes or typos here
def register_user(username: str = "User1"):
    new_key = f"pram_live_{uuid.uuid4().hex[:12]}"
    with sqlite3.connect("pram_service.db") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (api_key TEXT PRIMARY KEY, username TEXT, ram_saved_gb REAL)")
        conn.execute("INSERT INTO users (api_key, username, ram_saved_gb) VALUES (?, ?, 0)", 
                     (new_key, username))
    return {"api_key": new_key}

# 3. STATS ROUTE (For the SDK to verify the key)
@app.get("/v1/stats")
def get_stats(x_api_key: str = Header(None)):
    with sqlite3.connect("pram_service.db") as conn:
        cursor = conn.cursor()
        user = cursor.execute("SELECT username, ram_saved_gb FROM users WHERE api_key=?", (x_api_key,)).fetchone()
    
    if not user:
        raise HTTPException(status_code=403, detail="Invalid API Key")
        
    return {"username": user[0], "total_ram_saved_gb": user[1]}