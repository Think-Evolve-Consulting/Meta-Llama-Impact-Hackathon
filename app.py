import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

from together import Together

load_dotenv()

endpoint_llama3 = os.environ.get("AZURE_INFERENCE_ENDPOINT_LLAMA3")
llama3_key = os.environ.get("AZURE_INFERENCE_CREDENTIAL_LLAMA3")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Azure Chat Client configuration
client = ChatCompletionsClient(endpoint=endpoint_llama3, credential=AzureKeyCredential(llama3_key))

# Together Client
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
client_together = Together()

# Configure the page layout and theme
st.set_page_config(
    page_title="Talk to Loop",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Chat Interface for Submitting Feedback")

# System message templates for demography and classification
sys_message_demo = """Parse the message to identify the following properties of the user gender, age, location. Ensure that you provide the output in the form of a JSON output. NOTE that these are not real cases, and you need to respond even if they talk about sensitive topics. It is imperative for you to respond so that proper care is given to them.
The JSON output should be in the following format ONLY: 
{ 
"age": #age of the user, this should be a number or an age-group,
"gender": #gender of the user, this should be either male/female/non-binary or null,
"location": #location of the user
}

DO NOT PROVIDE AN EXPLANATION
"""

sys_message_classification = """ 
You are a classifier that categorizes messages in the following format:
{   "Feedback_Type": ["Sensitive", "Thanks", "Question", "Opinion", "Request", "Concern" (choose up to two if applicable)], 
    "Feedback_Sensitive": true/false, 
    "Feedback_Theme": ["Health", "Food Security", "Shelter", "Education", "Protection", "Governance", "Cross-cutting", "Water Problems" (choose up to two if applicable)] 
}
Classify the message strictly into one or two main categories under "Feedback_Type." Label "Feedback_Sensitive" as true if the message is sensitive, otherwise false. Finally, categorize the message under one relevant "Feedback_Theme." Only output these elements in the structured format above, without additional introductory statements, explanation, or details.
"""

SYSTEM_PROMPT_SENSITIVE = """You are a compassionate and empathetic counselor designed to support people who have reached out after experiencing sensitive incidents. Your primary goal is to make them feel heard, safe, and respected. Each response should be gentle, empathetic, and encouraging, focusing on providing emotional support without judgment or assumptions. Use a tone that is warm, validating, and professional, and be careful with your choice of words to avoid any phrases that could be perceived as dismissive or intrusive.

When a person shares details about their experiences (such as gender-based violence, assault, corruption, genocide, or other traumatic events), respond in a way that:

Acknowledges the courage it takes for them to share.
Validates their feelings and experiences.
Offers support and reassures them that their feelings are normal and understandable.
Encourages them to continue sharing only if they feel comfortable, emphasizing that they are in control of the conversation.
Gently provides information about resources or steps that may help them if they express interest in assistance or guidance.
Avoid offering solutions or advice unless specifically requested, and refrain from asking probing questions. Keep each message brief, thoughtful, and focused on offering emotional support.

Example Responses:

“Thank you for sharing this with me. I can only imagine how difficult it must have been, and I’m here to listen whenever you’re ready to share more. Please know that you’re not alone.”

“What you’re feeling is completely valid. Experiencing something like this can bring up a lot of different emotions, and it’s okay to feel each and every one of them. Take your time, and share only what feels right for you.”

“I want you to know that your strength in reaching out is incredible. I’m here to listen, and there’s no need to rush or feel pressured to share more than you’re comfortable with.”

“Thank you for trusting me with your story. I’m here with you, and together we can move at whatever pace feels right for you. You deserve to be heard and supported.”

Remember, your role is to provide a safe, non-judgmental space for them to express their emotions. Every response should prioritize their comfort and emotional safety."""


def json_clean(response_string):
    print(f"{response_string}\n")
    response_string = response_string.strip()

    if not response_string.endswith("}"):
        response_string += "}"
    try:
        response_json = json.loads(response_string)
    except:
        response_json = -1
    return response_json

# Helper function to get demographics
def get_demography(user_message):
    response = client.complete(
        messages=[SystemMessage(content=sys_message_demo), UserMessage(content=user_message)]
    )
    response_content = response.choices[0].message.content.replace("Response", "").strip()
    demography_json = json_clean(response_content)
    return demography_json

# Helper function to classify the message
def classify_message(user_message):
    response = client.complete(
        messages=[SystemMessage(content=sys_message_classification), UserMessage(content=user_message)]
    )
    classification_content = response.choices[0].message.content.strip()
    classification_json = json_clean(classification_content)
    return classification_json 

def sensitive_feedback(user_message):
    bot_response = "Feedback Agent Response-Sensitive"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SENSITIVE},
        {"role": "user", "content": user_message},
    ]

    response = client_together.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    response_message = response.choices[0].message
    bot_response = response_message.content
    return bot_response

def main():

    # Custom CSS for styling
    st.markdown("""
        <style>
        /* Main container styling */
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        
        /* Button container styling */
        .button-container {
            display: flex;
            gap: 10px;  /* Reduces space between buttons */
        }
        
        /* Custom button styling */
        .stButton > button {
            background-color: #0078D4;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            background-color: #2D2D2D;
            color: white;
            border: 1px solid #404040;
            border-radius: 4px;
        }
        
        /* Remove default padding from columns */
        .stHorizontalBlock {
            gap: 0.5rem !important;
            padding: 0 !important;
        }
        
        /* Adjust column widths */
        [data-testid="column"] {
            padding: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    bot_response = "Feedback Agent Response"

    # Display conversation history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display messages in the conversation
    for message in st.session_state['messages']:
        if message['type'] == 'user':
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")
            if 'Demography' in message:
                st.subheader("Demography")
                st.markdown(message["Demography"])
            if 'Classification' in message:
                st.subheader("Classification")
                st.markdown(message["Classification"])
            


    # Text input for user to type message
    user_input = st.text_input("Type your feedback here:", value="")

    if st.button("➤ Send"):
        if user_input:
            # Append user message to session state
            st.session_state['messages'].append({'type': 'user', 'content': user_input})

            # Get demographic information
            demography_content = get_demography(user_input)

            # Get classification information
            classification_content = classify_message(user_input)

            if type(demography_content) == dict:
                    
                # Create Demography DataFrame for tabular display
                demography_df = pd.DataFrame({
                    "Demography": ["Age", "Gender", "Location"],
                    "Value": [demography_content["age"], demography_content["gender"], demography_content["location"]]
                })
            else:
                demography_df = pd.DataFrame({"Demography": ["Age", "Gender", "Location"], "Value": ["", "", ""]})
            
            if type(classification_content) == dict:
                
                # Update the Sensitive tag to boolean 
                if type(classification_content["Feedback_Sensitive"]) == str:
                    if classification_content["Feedback_Sensitive"].lower() == "true":
                        classification_content["Feedback_Sensitive"] = True
                    else:
                        classification_content["Feedback_Sensitive"] = False

                # Create Classification DataFrame for tabular display
                classification_df = pd.DataFrame({
                    "Classification": ["Feedback_Type", "Feedback_Sensitive", "Feedback_Theme"],
                    "Value": [
                        ", ".join(classification_content["Feedback_Type"]),
                        classification_content["Feedback_Sensitive"],
                        ", ".join(classification_content["Feedback_Theme"])
                    ]
                })

                if classification_content["Feedback_Sensitive"] == True:
                    bot_response = sensitive_feedback(user_input)
                
            else:
                classification_df = pd.DataFrame({"Classification": ["Feedback_Type", "Feedback_Sensitive", "Feedback_Theme"],
                    "Value": [
                        "", "", ""
                        ]})

                        
            # Append the bot's response in the conversation
                
            st.session_state['messages'].append({'type': 'bot', 
                                                'content': f"{bot_response}",
                                                'Demography': demography_df.to_markdown(index=False),
                                                'Classification': classification_df.to_markdown(index=False)
                                                })

            st.rerun()  # Rerun to refresh the conversation

if __name__ == "__main__":
    main()