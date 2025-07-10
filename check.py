import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyBHcfR-FQf5JHkxK0Qmoe_mZkHynmEHHJU"))

model = genai.GenerativeModel("gemini-1.5-flash")

try:
    response = model.generate_content("Say something positive in English.")
    print("✅ Gemini says:", response.text)
except Exception as e:
    print("🔥 Test failed:", e)
    
