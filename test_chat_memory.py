import os
from fastapi.testclient import TestClient
from main import app
from database import upsert_user

def run_test():
    with TestClient(app) as client:
        print("Creating mock user profile with memory...")
        upsert_user({
            "user_id": "test_memory_user",
            "name": "Test User",
            "college": "Test College",
            "weak_topics": ["Fourier Series", "Thermodynamics"],
            "strong_topics": ["Data Structures"],
            "last_studied": "Binary Trees"
        })
        
        print("Sending chat request...")
        response = client.post(
            "/chat",
            json={
                "message": "What should I study today based on my profile?",
                "college": "Test College",
                "semester": "5",
                "department": "CS",
                "user_id": "test_memory_user",
                "chat_history": []
            }
        )
        
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except:
            print(f"Response Text: {response.text}")

if __name__ == "__main__":
    run_test()
