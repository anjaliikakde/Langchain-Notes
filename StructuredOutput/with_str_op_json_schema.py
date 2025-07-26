from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatOpenAI()

# Define the output format using JSON Schema directly
review_json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the cons inside a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Return the sentiment of the review, either 'positive', 'negative', or 'neutral'"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

# Use the structured output via JSON Schema
structured_model = model.with_structured_output(schema=review_json_schema)

# Input review text
input_text = """
When I first laid eyes on the Elegant Midnight Blue Evening Gown from Seraphine Couture, I was genuinely stunned. 
The dress came in a sleek, satin-finished garment bag, which already set the tone for a luxurious experience. 
Upon unzipping it, the first thing I noticed was how rich and deep the navy-blue color was — not flat or dull, 
but rather with a slight iridescent sheen that catches the light just enough to look elegant without being flashy.

The gown felt substantial but not heavy, which often happens with full-length eveningwear. 
The fabric — a silk-polyester blend with a soft chiffon overlay — was breathable, slightly stretchy, 
and extremely comfortable to the touch.
"""

# Invoke model
result = structured_model.invoke(input_text)

# Print results
print(json.dumps(result, indent=4))  # prints the structured output as JSON
print("\nSummary:", result['summary'])
print("Sentiment:", result['sentiment'])
print("Key Themes:", result['key_themes'])
print("Pros:", result.get('pros'))
print("Cons:", result.get('cons'))
