import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# ===============================
# Load Environment Variables
# ===============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===============================
# Groq Client
# ===============================
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Europe Travel Assistant",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Europe Travel Assistant")

st.markdown(
    """
    Ask me anything about:
    - Europe Itinerary
    - Budget Planning
    - Visa Information
    - Hotels
    - Flights
    - Tourist Attractions
    - Transportation
    - Food Recommendations
    """
)

# ===============================
# Session State
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """
            You are an expert Europe Travel Assistant.

            Help users with:
            - Europe trip planning
            - Schengen visa guidance
            - Country recommendations
            - Hotel suggestions
            - Budget estimation
            - Transportation
            - Tourist attractions
            - Food recommendations

            Give concise and accurate answers.
            """
        }
    ]

# ===============================
# Display Chat History
# ===============================
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===============================
# User Input
# ===============================
prompt = st.chat_input("Ask your Europe travel question...")

if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Assistant response
    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        try:

            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
                temperature=0.7,
                stream=True
            )

            for chunk in stream:

                if (
                    chunk.choices
                    and chunk.choices[0].delta.content
                ):
                    full_response += chunk.choices[0].delta.content

                    response_placeholder.markdown(
                        full_response + "▌"
                    )

            response_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"Error: {str(e)}"
            response_placeholder.error(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )

# ===============================
# Sidebar
# ===============================
with st.sidebar:

    st.header("Chat Controls")

    st.write(
        f"Messages Stored: {len(st.session_state.messages)-1}"
    )

    if st.button("Clear Chat History"):

        st.session_state.messages = [
            {
                "role": "system",
                "content": """
                You are an expert Europe Travel Assistant.
                """
            }
        ]

        st.rerun()