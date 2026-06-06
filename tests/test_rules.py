import os
from dotenv import load_dotenv
load_dotenv()
import vertexai
from rules_engine import evaluate_note_quality

try:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project:
        vertexai.init(project=project, location="us-central1")
        
    dummy_text = "Thermodynamics is the study of heat and work. The first law states that energy cannot be created or destroyed. Here is an example: A gas expands in a cylinder doing 50J of work while absorbing 100J of heat. The change in internal energy is 50J."
    print("Testing evaluate_note_quality...")
    result = evaluate_note_quality(dummy_text)
    print("Score:", result["score"])
    print("Strengths:", result["strengths"])
    print("Weaknesses:", result["weaknesses"])
    print("ALL TESTS PASSED")
except Exception as e:
    print("Error:", e)
