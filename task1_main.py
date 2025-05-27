"""
GROOT - An Open Source Chatbot using LangChain and Hugging Face Models
This chatbot demonstrates the use of:
- Streamlit for the web interface
- LangChain for conversation management and prompt handling
- Hugging Face's open source language models
- Structured output parsing for consistent responses
"""

import streamlit as st
from langchain.chains import ConversationChain  # For managing conversation flow
from langchain.memory import ConversationBufferMemory  # For maintaining chat history
from langchain.llms import HuggingFaceHub  # For accessing Hugging Face models
from langchain.output_parsers import ResponseSchema, StructuredOutputParser  # For structured outputs
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate  # For template management
import os  # For environment variables

# Setting up the Streamlit web interface
st.title(" GROOT")
st.write("Hello! I'm Groot I am here to answer your queries")

# Define how we want the bot's responses to be structured
# This helps ensure consistent and well-organized replies
response_schemas = [
    ResponseSchema(
        name="answer",  # The main part of the response
        description="The main response to the user's question"
    ),
    ResponseSchema(
        name="suggestions",  # Optional follow-up suggestions
        description="One or two short follow-up suggestions to keep the conversation flowing"
    )
]

# Create a parser that will help format the model's output according to our schema
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Define the template for how we'll talk to the model
# This template includes:
# - Instructions for the model
# - Format instructions from our parser
# - The user's input
# - Previous conversation history for context
prompt_template = """
Give a clear and direct response to the user's input.
{format_instructions}

User's input: {user_input}

Previous conversation context:
{history}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Initialize the session state to store our conversation
# Streamlit reruns the script on each interaction, so we need to store state
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Store all chat messages

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None  # Store the chatbot instance

# First time setup - handle API key and model selection
# This only runs when the user first loads the page
if 'huggingface_key' not in st.session_state:
    st.write("Get your free API key from huggingface.co")
    
    # Get API key
    api_key = st.text_input("Enter your HuggingFace API Key:", type="password")
    
    # Model selection
    model = st.selectbox("Choose a model:", [
        "google/flan-t5-small",     # Small, fast model
        "facebook/opt-125m",        # Small but capable model
        "bigscience/bloom-560m"     # Medium sized model
    ], help="Smaller models are faster but less capable")
    
    if api_key:
        try:
            # Save API key
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            st.session_state.huggingface_key = api_key
            
            # Create chatbot
            llm = HuggingFaceHub(
                repo_id=model,
                model_kwargs={"temperature": 0.7}
            )
            
            # Add memory
            memory = ConversationBufferMemory()
            st.session_state.chatbot = ConversationChain(
                llm=llm,
                memory=memory
            )
            
            st.success("Ready to chat! 🚀")
            
        except Exception as e:
            st.error("Oops! Check your API key and try again!")

# Display the chat history
# Loop through all messages and display them with appropriate formatting
for message in st.session_state.messages:
    if message["role"] == "user":
        # Display user messages
        st.write("You:", message["content"])
    else:
        try:
            # For bot messages, we need to handle both parsed and unparsed responses
            # If the content is a string, it needs to be parsed into our structured format
            if isinstance(message["content"], str):
                parsed = parser.parse(message["content"])
            else:
                # If it's already parsed, use it as is
                parsed = message["content"]
                
            # Show answer
            st.write("Bot:", parsed.answer)
            
            # Show suggestions in a small expandable section
            if hasattr(parsed, 'suggestions'):
                with st.expander("Suggestions"):
                    st.write(parsed.suggestions)
        except:
            # Fallback to simple display if parsing fails
            st.write("Bot:", message["content"])

# Handle user input and generate responses
# Only show input if user has provided an API key
if st.session_state.get('huggingface_key'):
    # Get user's message through text input
    message = st.text_input("Your message:", key="user_input")
    
    if message:
        # Store the user's message in chat history
        st.session_state.messages.append({
            "role": "user",
            "content": message
        })
        
        try:
            # Get bot response with format instructions
            response = st.session_state.chatbot.predict(
                input=message,
                format_instructions=parser.get_format_instructions()
            )
            
            # Parse the response
            parsed_response = parser.parse(response)
            
            # Add bot message
            st.session_state.messages.append({
                "role": "bot",
                "content": parsed_response
            })
            
            # Refresh display
            st.experimental_rerun()
            
        except Exception as e:
            st.error("Sorry! Try again!")

    # Clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chatbot.memory.clear()
        st.experimental_rerun()
