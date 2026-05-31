import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(
    project="notely-study-assistant",
    location="us-central1"
)

model = GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Say hello")
print(response.text)
