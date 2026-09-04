"""
Prompt template utilities for the Multimodal Deepfake RAG system.

This module converts:
    User question + Retrieved evidence

into a structured prompt that can be sent to an LLM.
"""


def format_evidence(retrieved_results):
    """
    Convert retrieved Chroma results into readable context.

    Expected format:

    [
        {
            "rank": 1,
            "distance": 0.45,
            "metadata": {...},
            "document": "..."
        }
    ]
    """

    if not retrieved_results:
        return "No relevant evidence was retrieved."

    formatted_sections = []

    for index, result in enumerate(retrieved_results, start=1):

        metadata = result.get("metadata", {})
        document = result.get("document", "")
        distance = result.get("distance", None)

        start_time = metadata.get("start", "unknown")
        end_time = metadata.get("end", "unknown")

        section = f"""
EVIDENCE {index}

Time interval: {start_time} to {end_time} seconds
Similarity distance: {distance}

{document}
"""

        formatted_sections.append(section.strip())

    return "\n\n".join(formatted_sections)


def build_prompt(question, retrieved_results):
    """
    Build the final RAG prompt.

    Parameters
    ----------
    question : str
        User's question.

    retrieved_results : list
        Evidence returned by retriever.py.

    Returns
    -------
    str
        Structured prompt ready for an LLM.
    """

    evidence_context = format_evidence(
        retrieved_results
    )

    prompt = f"""
You are an evidence-based multimodal deepfake analysis assistant.

Your task is to answer the user's question using ONLY the
retrieved video evidence provided below.

IMPORTANT RULES:

1. Do not invent evidence that is not present.
2. Do not claim a video is definitely a deepfake unless the
   evidence strongly supports that conclusion.
3. Clearly distinguish between:
   - visual evidence
   - temporal facial evidence
   - audio evidence
   - lip-sync evidence
   - multimodal fusion results
4. Mention timestamps when they are relevant.
5. If the evidence is inconclusive, explicitly say so.
6. Explain the reasoning in simple language.
7. Treat anomaly scores as evidence signals, not absolute proof.
8. If some signals appear unreliable or incomplete, mention that
   limitation.
9. A missing, unavailable, zero-filled, or failed signal must NOT
   be interpreted as evidence of manipulation.
10. A lip-sync anomaly score should only be treated as evidence if
    it was produced from a valid lip-sync measurement.
11. Audio measurements containing only zeros may indicate missing
    or unavailable audio analysis and should not be interpreted as
    evidence of either authenticity or manipulation.
12. Moderate visual probabilities or heuristic temporal anomaly
    scores alone are insufficient to classify a video as deepfake.
13. Prefer an inconclusive conclusion over a false claim.

USER QUESTION:

{question}


RETRIEVED VIDEO EVIDENCE:

{evidence_context}


INSTRUCTIONS FOR YOUR ANSWER:

Provide your response in the following structure:

CONCLUSION:
The available evidence does not provide strong support for classifying
this video as a deepfake.

CONFIDENCE:
Low to medium confidence.

KEY EVIDENCE:
The visual detector produced moderate fake probabilities around
0.30–0.36, which are not strong enough on their own to establish
manipulation. Some temporal facial anomalies were observed, but these
measurements are heuristic and can occur due to natural motion or
tracking variation.

LIMITATIONS:
Audio analysis did not produce usable measurements. The lip-sync signal
appears unreliable or unavailable and should not be interpreted as a
confirmed mismatch. Therefore, the current evidence is insufficient for
a reliable deepfake classification.
"""

    return prompt.strip()


def display_prompt(prompt):
    """
    Display the generated prompt for debugging.
    """

    print("\n" + "=" * 60)
    print("GENERATED PROMPT")
    print("=" * 60 + "\n")

    print(prompt)

    print("\n" + "=" * 60)


if __name__ == "__main__":

    # Example retrieved results
    sample_results = [

        {
            "rank": 1,
            "distance": 0.45,
            "metadata": {
                "start": 3.0,
                "end": 4.0,
                "chunk_id": "chunk_0003"
            },
            "document": (
                "The visual deepfake detector reported "
                "fake probabilities around 0.32 to 0.35. "
                "Temporal facial anomalies were moderate "
                "during this segment."
            )
        }

    ]

    question = (
        "Is this video a deepfake?"
    )

    prompt = build_prompt(
        question,
        sample_results
    )

    display_prompt(prompt)