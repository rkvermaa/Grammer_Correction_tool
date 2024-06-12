import openai
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import json

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Read the system prompt from the file
with open('system_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read()

# Define the prompt template
PROMPT_TEMPLATE = """
System: {system}
Input Paragraph: "{input_paragraph}"
"""

# Create a PromptTemplate object
prompt = PromptTemplate(
    input_variables=["system", "input_paragraph"],
    template=PROMPT_TEMPLATE
)

# Create the ChatOpenAI object
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")  # or use "gpt-4"

# Create the LLM chain
chain = LLMChain(prompt=prompt, llm=llm)

# Define the input model for the FastAPI endpoint
class ParagraphInput(BaseModel):
    paragraph: str

# Initialize FastAPI
app = FastAPI()

@app.post("/correct_grammar")
async def correct_grammar(input: ParagraphInput):
    try:
        # Generate the correction
        result = chain.run({
            "system": SYSTEM_PROMPT,
            "input_paragraph": input.paragraph
        })
        return {"result": json.loads(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for running the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
