import json
import streamlit as st
from dotenv import load_dotenv
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
import pandas as pd

load_dotenv()
endpoint_llama3 = os.environ.get("AZURE_INFERENCE_ENDPOINT_LLAMA3")
llama3_key = os.environ.get("AZURE_INFERENCE_CREDENTIAL_LLAMA3")

# Azure Chat Client configuration
client = ChatCompletionsClient(endpoint=endpoint_llama3, credential=AzureKeyCredential(llama3_key))

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
    user_input = st.text_input("Type your message here:", value="")

    if st.button("Send Text"):
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
                    
                # Create Classification DataFrame for tabular display
                classification_df = pd.DataFrame({
                    "Classification": ["Feedback_Type", "Feedback_Sensitive", "Feedback_Theme"],
                    "Value": [
                        ", ".join(classification_content["Feedback_Type"]),
                        classification_content["Feedback_Sensitive"],
                        ", ".join(classification_content["Feedback_Theme"])
                    ]
                })
            else:
                classification_df = pd.DataFrame({"Classification": ["Feedback_Type", "Feedback_Sensitive", "Feedback_Theme"],
                    "Value": [
                        "", "", ""
                        ]})

                        
            # Append the bot's response in the conversation
                
            st.session_state['messages'].append({'type': 'bot', 
                                                'content': "Feedback Agent Response",
                                                'Demography': demography_df.to_markdown(index=False),
                                                'Classification': classification_df.to_markdown(index=False)
                                                })

            st.rerun()  # Rerun to refresh the conversation

if __name__ == "__main__":
    main()