import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,  ValidationError, Field
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from typing import List, Dict, Any, Optional
import json


# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI
app = FastAPI()

# Define Pydantic models for the output
class CorrectionOutput(BaseModel):
    revised_text: str = Field(default=None, description="It is the suggested replacement in sentence")
    comment: str =Field(default=None, description="it is description of the suggestion.")
    type: int = Field(default=None, description="Major type of the error. Here, 1 denotes grammar error, 2 spelling, 3 advisor, 4 enhancement, and 5 style guide.")
    cta_present: bool = Field(default=None, description="Indicates if a call-to-action (CTA) for the correction is present. If the value is false, the revised_text key will be an empty string.")
    error_id: str = Field(default=None, description="Unique identifier for the error.")
    error_category: str = Field(default=None, description="Category of the error, for example, Articles.")

class SentenceResult(BaseModel):
    start_index: int = Field(default=None, description="Identify the starting index of a particular word within the overall context of the sentence.")
    end_index: int = Field(default=None, description="Identify the ending index of a particular word within the overall context of the sentence.")
    covered_text: str = Field(default=None, description="The text within the sentence, as indicated by the start and end indices, for which Language model suggested has suggesed a revision.")
    output: List[CorrectionOutput] = Field(default=None, description="The array of multiple suggestions for the covered_text.")

class SentenceResponse(BaseModel):
    sentence: str = Field(default=None, description="This key contains the sentence text.")
    start_index: int = Field(default=None, description="Identify the starting index of a particular sentence within the overall context of the paragraph.")
    end_index: int = Field(default=None, description="Identify the ending index of a particular sentence within the overall context of the paragraph.")
    sentence_result: List[SentenceResult] = Field(default=None, description="This is a list containing details of all errors identified in the sentence.")

class CorrectionResponse(BaseModel):
    status: bool = Field(default=None, description="This denotes a Boolean value, indicating whether the request processing was successful (status: true) or failed (status: false).")
    message: str = Field(default=None, description="A message describing the outcome of the operation.")
    input: str = Field(default=None, description="This is the text to be processed sent in the request.")
    response: List[SentenceResponse] = Field(default=None, description="A list of sentences with Grammar Error Correction responses.")
    language: str = Field(default=None, description="Indicates the language used in the input text.")
    style_guide: Optional[str] = Field(default=None, description="Indicates the style guide present in the input request.")



input_paragraph = "step lader approach gradualy escalate diet therapy. This aproach has been creted keepiing sustainability in mind with simpler and traditional diets at start and increasingly restrictive diets at subsequent escalations. Diet ladder is being successsfully used at oour practice for the last two years with good patient adherence and satisfaction. Numerous meta-analyses have tried compare diet and ascertain the best diet, but all diets seem to work almost equally in the long term in unselected populations (Fogelholm, Anderssen, Gunnarsdottir, Lahti-Koski, 2012). Therefore, we believe, each individual would respond positively to one of the dietary approaches and the diet ladder lays down a practical framework to find that approach."

parser = PydanticOutputParser(pydantic_object=CorrectionResponse)
# Create the ChatOpenAI object
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4-turbo-2024-04-09") # older model gpt-3.5-turbo

prompt = PromptTemplate(
    template="You are an advanced language model like Trinka capable of correcting grammar and identifying spelling errors in a given text. For the provided paragraph, please:\
            1. Correct each sentence of the given text grammatically.\
            2. Do not remove or add any word unnecessarily.\
            3. Identify misspelled words.\
            4. For each misspelled word, provide its corrected word and its starting index and ending index..\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

class InputParagraph(BaseModel):
    paragraph: str

@app.post("/correct_grammar")
async def correct_paragraph(input: InputParagraph):
    try:
        result = chain.invoke({"query": input.paragraph})
        return json.loads(result.model_dump_json())
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.json())

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
