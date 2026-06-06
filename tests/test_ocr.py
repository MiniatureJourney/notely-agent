import os
from dotenv import load_dotenv
load_dotenv()
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import json

try:
    print("Initializing Gemini Model...")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project:
        vertexai.init(project=project, location="us-central1")
        
    model = GenerativeModel("gemini-2.5-flash")
    
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
    
    print("Gemini Response:", response.text)
    
    import main
    print("main.py imported successfully!")
    print("ALL TESTS PASSED")
except Exception as e:
    print(f"Error: {e}")
