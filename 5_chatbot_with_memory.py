"""
Building a Chatbot with Memory using LangChain
Create a simple chatbot that can remember previous interactions using LangChain's memory components. Implement conversation memory to make your chatbot maintain context throughout a conversation.
Instructions:
  1. Import the necessary components for chat history and conversation memory.
  2. Set up a language model for your chatbot.
  3. Create a conversation chain with memory capabilities.
  4. Implement a simple interactive chat interface.
  5. Test the memory capabilities with a series of related questions.
  6. Examine how the conversation history is stored and accessed.
"""


from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

# 1. Set up the language model
model_id = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.2,
}
credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
project_id = "skills-network"

# Initialize the model
model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
llm = WatsonxLLM(model=model)

# 2. Create a simple conversation with chat history
history = ChatMessageHistory()

# Add some initial messages
history.add_user_message("Hello, my name is Alice.")
history.add_ai_message("Hello Alice! It's nice to meet you. How can I help you today?")

# 3. Print the current conversation history
print("Initial Chat History:")
for message in history.messages:
    sender = "Human" if isinstance(message, HumanMessage) else "AI"
    print(f"{sender}: {message.content}")

# 4. Set up a conversation chain with memory
memory = ConversationBufferMemory(chat_memory=history)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 5. Function to simulate a conversation
def chat_simulation(conversation, inputs):
    """Run a series of inputs through the conversation chain and display responses"""
    print("\n=== Beginning Chat Simulation ===")
    
    for i, user_input in enumerate(inputs):
        print(f"\n--- Turn {i+1} ---")
        print(f"Human: {user_input}")
        
        # Get response from the conversation chain
        response = conversation.invoke(input=user_input)
        
        # Print the AI's response
        print(f"AI: {response['response']}")
    
    print("\n=== End of Chat Simulation ===")

# 6. Test with a series of related questions
test_inputs = [
    "My favorite color is blue.",
    "I enjoy hiking in the mountains.",
    "What activities would you recommend for me?",
    "What was my favorite color again?",
    "Can you remember both my name and my favorite color?"
]

chat_simulation(conversation, test_inputs)

# 7. Examine the conversation memory
print("\nFinal Memory Contents:")
print(conversation.memory.buffer)

# 8. Create a new conversation with a different type of memory (optional)
from langchain.memory import ConversationSummaryMemory

# Create a summarizing memory that will compress the conversation
summary_memory = ConversationSummaryMemory(llm=llm)
# Save the initial context to the summary memory
summary_memory.save_context(
    {"input": "Hello, my name is Alice."}, 
    {"output": "Hello Alice! It's nice to meet you. How can I help you today?"}
)
summary_conversation = ConversationChain(
   llm=llm,
   memory=summary_memory,
   verbose=True
)

print("\\\\\\\\n\\n=== Testing Conversation Summary Memory ===")
# Let's use the same inputs for comparison
chat_simulation(summary_conversation, test_inputs)

print("\\nFinal Summary Memory Contents:")
print(summary_memory.buffer)

# 9. Compare the two memory types
print("\n=== Memory Comparison ===")
print(f"Buffer Memory Size: {len(conversation.memory.buffer)} characters")
print(f"Summary Memory Size: {len(summary_memory.buffer)} characters")
print("\nThe conversation summary memory typically creates a more compact representation of the chat history.")