import cohere

# Replace with your actual Cohere API key
co = cohere.Client("key")

try:
    response = co.generate(
        model='command-r-plus',
        prompt='Hello! How are you feeling today?',
        max_tokens=50
    )
    print("✅ Cohere Response:\n", response.generations[0].text.strip())
except Exception as e:
    print("❌ Cohere Error:", e)
