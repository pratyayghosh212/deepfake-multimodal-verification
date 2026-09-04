"""
LLM interaction module for the
Multimodal Deepfake Evidence RAG system.

This module sends a structured prompt
to the Groq LLM API and returns
an evidence-based explanation.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# =========================================================
# Configuration
# =========================================================




# =========================================================
# Environment setup
# =========================================================

load_dotenv()


def get_api_key():
    """
    Load the Groq API key from environment variables.
    """

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "\nGROQ_API_KEY was not found.\n"
            "Create a .env file in the project root "
            "and add:\n\n"
            "GROQ_API_KEY=your_api_key_here\n"
        )

    return api_key


# =========================================================
# Groq client
# =========================================================

def create_client():
    """
    Create and return the Groq client.
    """

    api_key = get_api_key()

    client = Groq(
        api_key=api_key
    )

    return client


# =========================================================
# LLM interaction
# =========================================================

def generate_response(
    prompt,
    temperature=0.2,
    max_tokens=1000
):
    """
    Send a prompt to the Groq LLM.

    Parameters
    ----------
    prompt : str
        Structured RAG prompt.

    temperature : float
        Controls randomness.
        Lower values produce more
        consistent answers.

    max_tokens : int
        Maximum response length.

    Returns
    -------
    str
        LLM-generated response.
    """

    client = create_client()

    print(
        f"\nSending request to Groq model: "
        f"{MODEL_NAME}"
    )

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful multimodal "
                    "deepfake evidence analysis assistant. "
                    "You must base conclusions only on "
                    "the evidence provided by the user."
                )
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=temperature,

        max_completion_tokens=max_tokens
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return answer


# =========================================================
# Display utilities
# =========================================================

def display_response(answer):
    """
    Display the LLM response.
    """

    print("\n" + "=" * 60)
    print("LLM ANALYSIS")
    print("=" * 60 + "\n")

    print(answer)

    print("\n" + "=" * 60)


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    test_prompt = """
You are analyzing multimodal evidence from a video.

USER QUESTION:

Is this video likely to be a deepfake?


RETRIEVED EVIDENCE:

Between 3.0 and 4.0 seconds:

- Visual fake probability was approximately 0.32 to 0.35.
- Temporal facial anomalies were moderate.
- Multimodal evidence levels ranged from low to medium.

Audio evidence was incomplete.

Provide an evidence-based answer.

Use the following format:

CONCLUSION:
CONFIDENCE:
KEY EVIDENCE:
LIMITATIONS:
"""

    print(
        "\nInitializing Groq LLM..."
    )

    answer = generate_response(
        test_prompt
    )

    display_response(answer)