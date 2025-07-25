from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    
    key_theams : Annotated[list[str], "Write done all the key theams discussed in the review in a list"]
    pros : Annotated[Optional[list[str]], " Write down all the prons inside a list "]
    cons : Annotated[Optional[list[str]], " Write down all the cons inside a list "]
    summary : Annotated[str, "A brief summary of the review"]
    sentiment : Annotated[str, "Return the sentiment of the review, either 'positive', 'negative', or 'neutral'"]
    
structured_model = model.with_structured_output(Review)   
    
result = structured_model.invoke("""When I first laid eyes on the Elegant Midnight Blue Evening Gown from Seraphine Couture, I was genuinely stunned. The dress came in a sleek, satin-finished garment bag, which already set the tone for a luxurious experience. Upon unzipping it, the first thing I noticed was how rich and deep the navy-blue color was — not flat or dull, but rather with a slight iridescent sheen that catches the light just enough to look elegant without being flashy.

The gown felt substantial but not heavy, which often happens with full-length eveningwear. The fabric — a silk-polyester blend with a soft chiffon overlay — was breathable, slightly stretchy, and extremely comfortable to the touch.""")

print(result)
print(result['summary'])
print(result['sentiment'])