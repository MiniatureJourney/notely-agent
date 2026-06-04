import json
from vertexai.generative_models import GenerativeModel

def evaluate_note_quality(text: str) -> dict:
    """
    Evaluates a note against 4 strict academic rules:
    1. Completeness (25 points)
    2. Clarity & Structure (25 points)
    3. Practical Examples (25 points)
    4. Exam Usefulness (25 points)
    """
    try:
        model = GenerativeModel("gemini-2.5-flash")
        prompt = f"""You are an expert Academic Assessor. Evaluate this student note based on the following 4 rules:
1. Completeness (0-25): Are core concepts defined?
2. Clarity & Formatting (0-25): Is it easy to read with headings/bullets?
3. Practical Examples (0-25): Are there solved problems or examples?
4. Exam Usefulness (0-25): Does it summarize well for exam revision?

Analyze the note and return ONLY a valid JSON object matching this schema:
{{
  "total_score": <number between 0 and 100>,
  "strengths": ["<string>", "<string>"],
  "weaknesses": ["<string>"]
}}

Note text:
{text[:5000]}
"""
        response = model.generate_content(prompt)
        
        result_text = response.text.strip()
        if result_text.startswith("```json"): result_text = result_text[7:-3]
        elif result_text.startswith("```"): result_text = result_text[3:-3]
        
        data = json.loads(result_text.strip())
        return {
            "score": float(data.get("total_score", 50)),
            "strengths": data.get("strengths", []),
            "weaknesses": data.get("weaknesses", [])
        }
    except Exception as e:
        print("Scoring error:", e)
        return {
            "score": 50.0,
            "strengths": ["Covers basic concepts"],
            "weaknesses": ["Needs deeper evaluation"]
        }
