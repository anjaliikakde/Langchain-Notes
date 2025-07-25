from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    key_themes: List[str] = Field(description="Write down all the key themes discussed in the review in a list")
    pros: Optional[List[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[List[str]] = Field(default=None, description="Write down all the cons inside a list")
    summary: str = Field(description="A brief summary of the review")          
    sentiment: str = Field(description="Return the sentiment of the review, either 'positive', 'negative', or 'neutral'")

structured_model = model.with_structured_output(Review)   
    
result = structured_model.invoke("""When I first laid eyes on the Elegant Midnight Blue Evening Gown from Seraphine Couture, I was genuinely stunned. The dress came in a sleek, satin-finished garment bag, which already set the tone for a luxurious experience. Upon unzipping it, the first thing I noticed was how rich and deep the navy-blue color was — not flat or dull, but rather with a slight iridescent sheen that catches the light just enough to look elegant without being flashy.

The gown felt substantial but not heavy, which often happens with full-length eveningwear. The fabric — a silk-polyester blend with a soft chiffon overlay — was breathable, slightly stretchy, and extremely comfortable to the touch.""")

print(result)
print(result.summary)
print(result.sentiment) 
print(result.key_themes)  # Fixed typo: was 'key_theams'
print(result.pros)
print(result.cons)