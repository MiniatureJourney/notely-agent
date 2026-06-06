import os
import json
from dotenv import load_dotenv

load_dotenv()
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Local imports
from database import _db, insert_note
from rules_engine import evaluate_note_quality

print("=== STARTING COMPREHENSIVE TESTS ===")

# Test 1: MongoDB Connection
try:
    print("\n1. Testing MongoDB Connectivity...")
    _db().command('ping')
    print("   [PASS] MongoDB is connected successfully!")
except Exception as e:
    print(f"   [FAIL] MongoDB error: {e}")

# Test 2: Vertex AI / Gemini Initialization
try:
    print("\n2. Testing Vertex AI & Gemini...")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project:
        vertexai.init(project=project, location="us-central1")
    model = GenerativeModel("gemini-2.5-flash")
    print("   [PASS] Gemini Model Initialized successfully!")
except Exception as e:
    print(f"   [FAIL] Gemini init error: {e}")

# Test 3: Rules Engine Evaluation
try:
    print("\n3. Testing Rule-Based Note Scoring Engine...")
    dummy_text = "Operating Systems. A process is a program in execution. It contains the program code and its current activity. Examples include Windows, Linux, and macOS."
    result = evaluate_note_quality(dummy_text)
    print(f"   [INFO] Score received: {result['score']}")
    if 'strengths' in result and 'weaknesses' in result:
        print("   [PASS] Rules engine successfully generated score and feedback!")
    else:
        print("   [FAIL] Rules engine missing required schema keys.")
except Exception as e:
    print(f"   [FAIL] Rules engine error: {e}")

# Test 4: OCR Image Parse Simulation
try:
    print("\n4. Testing OCR Parse Simulation...")
    # 1x1 white pixel PNG
    dummy_png_bytes = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")
    prompt = '''Analyze this handwritten or typed note.
Extract the raw text comprehensively.
Also, auto-classify it into a Subject, Topic (chapter), Semester (1-8), and an appropriate Title.
Respond ONLY in valid JSON format exactly like this:
{
  "content": "extracted text...",
  "subject": "e.g. Engineering Mathematics",
  "chapter": "e.g. Fourier Series",
  "semester": 5,
  "title": "e.g. Fourier Transforms Overview"
}'''

    response = model.generate_content([
        Part.from_data(data=dummy_png_bytes, mime_type="image/png"),
        prompt
    ])
    
    text = response.text.strip()
    if text.startswith("```json"): text = text[7:-3]
    elif text.startswith("```"): text = text[3:-3]
    
    parsed = json.loads(text.strip())
    print(f"   [INFO] OCR Extracted Title: {parsed.get('title')}")
    print("   [PASS] OCR Simulation completed and correctly formatted!")
except Exception as e:
    print(f"   [FAIL] OCR Simulation error: {e}")

print("\n=== ALL TESTS FINISHED ===")
