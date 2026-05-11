"""
Creating and Using a JSON Output Parser
A JSON output parser to structure the responses from your LLM.

Instructions:
- Import the necessary components to create a JSON output parser.
- Create a prompt template that requests information in JSON format (hint: use the provided template).
- Build a chain that connects your prompt, LLM, and JSON parser.
- Test your parser using at least three different inputs.
- Access and display specific fields from the parsed JSON output.
- Verify that your output is properly structured and accessible as a Python dictionary.

!pip install --force-reinstall --no-cache-dir tenacity==8.2.3 --user
!pip install "ibm-watsonx-ai==1.0.8" --user
!pip install "ibm-watson-machine-learning==1.0.367" --user
!pip install "langchain-ibm==0.1.7" --user
!pip install "langchain-community==0.2.10" --user
!pip install "langchain-experimental==0.0.62" --user
!pip install "langchainhub==0.1.18" --user
!pip install "langchain==0.2.11" --user
!pip install "pypdf==4.2.0" --user
!pip install "chromadb==0.4.24" --user

restart kernel
import os
os._exit(00)

"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Create your JSON parser
json_parser = JsonOutputParser()

# Create more explicit format instructions
format_instructions = """RESPONSE FORMAT: Return ONLY a single JSON object—no markdown, no examples, no extra keys.  It must look exactly like:
{
  "title": "movie title",
  "director": "director name",
  "year": 2000,
  "genre": "movie genre"
}

IMPORTANT: Your response must be *only* that JSON.  Do NOT include any illustrative or example JSON."""

# Create your prompt template with clearer instructions
prompt_template = PromptTemplate(
    template="""You are a JSON-only assistant.

Task: Generate info about the movie "{movie_name}" in JSON format.

{format_instructions}
""",
    input_variables=["movie_name"],
    partial_variables={"format_instructions": format_instructions},
)

# Create the chain without cleaning step
movie_chain = prompt_template | llama_llm | json_parser

# Test with a movie name
movie_name = "The Matrix"
result = movie_chain.invoke({"movie_name": movie_name})

# Print the structured result
print("Parsed result:")
print(f"Title: {result['title']}")
print(f"Director: {result['director']}")
print(f"Year: {result['year']}")
print(f"Genre: {result['genre']}")