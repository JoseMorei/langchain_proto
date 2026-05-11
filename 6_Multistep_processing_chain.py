"""
Implementing Multi-Step Processing with Different Chain Approaches
Create a multi-step information processing system using both traditional chains and the modern LCEL approach. Build a system that analyzes product reviews, extracts key information, and generates responses based on the analysis.

Instructions:
- Import the necessary components for both traditional chains and LCEL.
- Implement a three-step process using both traditional SequentialChain and modern LCEL approaches.
- Create templates for sentiment analysis, summarization, and response generation.
- Test your implementations with sample product reviews.
- Compare the flexibility and readability of both approaches.
- Document the advantages and disadvantages of each method.
"""

from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Sample product reviews for testing
positive_review = """I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing. 
The built-in grinder saves me so much time in the morning, and the programmable timer means 
I wake up to fresh coffee every day. Worth every penny and highly recommended to any coffee enthusiast."""

negative_review = """Disappointed with this laptop. It's constantly overheating after just 30 minutes of use, 
and the battery life is nowhere near the 8 hours advertised - I barely get 3 hours. 
The keyboard has already started sticking on several keys after just two weeks. Would not recommend to anyone."""

# Step 1: Define the prompt templates for each processing step
sentiment_template = """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:
"""

summary_template = """Summarize the following product review into 3-5 key bullet points.
Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:
"""

response_template = """Write a helpful response to a customer based on their product review.
If the sentiment is positive, thank them for their feedback. If negative, express understanding 
and suggest a solution or next steps. Personalize based on the specific points they mentioned.

Review: {review}
Sentiment: {sentiment}
Key points: {summary}

Response to customer:
"""

# Create prompt templates for each step
sentiment_prompt = PromptTemplate.from_template(sentiment_template)
summary_prompt = PromptTemplate.from_template(summary_template)
response_prompt = PromptTemplate.from_template(response_template)


# PART 1: Traditional Chain Approach
# Create individual LLMChains for each step
sentiment_chain = LLMChain(
    llm=llama_llm, 
    prompt=sentiment_prompt, 
    output_key="sentiment"
)

summary_chain = LLMChain(
    llm=llama_llm, 
    prompt=summary_prompt, 
    output_key="summary"
)

response_chain = LLMChain(
    llm=llama_llm, 
    prompt=response_prompt, 
    output_key="response"
)

# Create a SequentialChain to connect all steps
traditional_chain = SequentialChain(
    chains=[sentiment_chain, summary_chain, response_chain],
    input_variables=["review"],
    output_variables=["sentiment", "summary", "response"],
    verbose=True
)


# PART 2: LCEL Approach
# Create individual chain components using the pipe operator (|)
sentiment_chain_lcel = sentiment_prompt | llama_llm | StrOutputParser()
summary_chain_lcel = summary_prompt | llama_llm | StrOutputParser()
response_chain_lcel = response_prompt | llama_llm | StrOutputParser()

# Connect the components using RunnablePassthrough.assign()
lcel_chain = (
    RunnablePassthrough.assign(
        sentiment=lambda x: sentiment_chain_lcel.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        summary=lambda x: summary_chain_lcel.invoke({
            "review": x["review"], 
            "sentiment": x["sentiment"]
        })
    )
    | RunnablePassthrough.assign(
        response=lambda x: response_chain_lcel.invoke({
            "review": x["review"], 
            "sentiment": x["sentiment"], 
            "summary": x["summary"]
        })
    )
)


# Test both implementations
def test_chains(review):
    """Test both chain implementations with the given review"""
    print("\n" + "="*50)
    print(f"TESTING WITH REVIEW:\n{review[:100]}...\n")
    
    print("TRADITIONAL CHAIN RESULTS:")
    traditional_results = traditional_chain.invoke({"review": review})
    print(f"Sentiment: {traditional_results['sentiment']}")
    print(f"Summary: {traditional_results['summary']}")
    print(f"Response: {traditional_results['response']}")
    
    print("\nLCEL CHAIN RESULTS:")
    lcel_results = lcel_chain.invoke({"review": review})
    print(f"Sentiment: {lcel_results['sentiment']}")
    print(f"Summary: {lcel_results['summary']}")
    print(f"Response: {lcel_results['response']}")
    
    print("="*50)

# Run tests
test_chains(positive_review)
test_chains(negative_review)
