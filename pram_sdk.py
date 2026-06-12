import requests
import os
import sys
from run_optimizer import main as run_engine

class PramClient:
    def __init__(self, api_key):
        self.api_key = "pram_live_eefdb95a1638"
        self.api_url = " http://127.0.0.1:8000"

    def optimize_model(self, model_path):
        print(f"📡 Connecting to PRAM Cloud...")
        
        # 1. AUTH CHECK
        headers = {"x-api-key": self.api_key}
        try:
            auth_res = requests.get(f"{self.api_url}/v1/stats", headers=headers)
            if auth_res.status_code != 200:
                print("❌ Access Denied: Invalid API Key.")
                return
            
            user_data = auth_res.json()
            print(f"🔓 Authenticated as {user_data['username']}. Starting engine...")

            # 2. RUN LOCAL ENGINE (This triggers the code we built earlier)
            # Capture the results for the dashboard
            from model.analyzer import analyze_model
            size_before = analyze_model(model_path)['model_size_gb']
            
            run_engine(model_path) # The original core logic
            
            size_after = analyze_model(model_path)['model_size_gb'] # Assuming name changed
            saved = size_before - size_after

            # 3. LOG USAGE BACK TO WEBSITE
            requests.post(
                f"{self.api_url}/v1/log-usage", 
                headers=headers, 
                json={"model_name": os.path.basename(model_path), "saved_gb": saved}
            )
            print(f"📊 Success! {round(saved, 2)} GB reduction logged to your dashboard.")

        except Exception as e:
            print(f"⚠️ Offline Mode or Server Error: {e}")

# --- PRODUCTION USAGE ---
if __name__ == "__main__":
    # The user enters the key they got from your dashboard
    client = PramClient(api_key="pram_live_PASTE_YOUR_KEY_HERE")
    client.optimize_model("models/hf_model")