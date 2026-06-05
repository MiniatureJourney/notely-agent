import os
from fastapi.testclient import TestClient
from main import app

def run_test():
    with TestClient(app) as client:
        print("Testing /api/ocr-note...")
        # Create a dummy image
        dummy_png_bytes = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")
        
        response = client.post(
            "/api/ocr-note",
            data={"uploaded_by": "test_user", "college": "Test College", "department": "CS"},
            files={"file": ("test.png", dummy_png_bytes, "image/png")}
        )
        
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except:
            print(f"Response Text: {response.text}")

if __name__ == "__main__":
    run_test()
