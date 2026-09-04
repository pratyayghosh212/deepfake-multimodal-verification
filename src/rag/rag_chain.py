

from retriever import (
    load_embedding_model,
    get_collection,
    embed_query,
    retrieve_evidence,
    format_results
)

from prompt_template import build_prompt

from llm import (
    generate_response,
    display_response
)


# =========================================================
# RAG Pipeline
# =========================================================

def run_rag_pipeline(
    question,
    top_k=3
):
    """
    Run the complete RAG pipeline.

    Parameters
    ----------
    question : str
        User's question about the video.

    top_k : int
        Number of evidence chunks to retrieve.

    Returns
    -------
    str
        Final LLM-generated answer.
    """

    print("\n" + "=" * 60)
    print("MULTIMODAL DEEPFAKE RAG PIPELINE")
    print("=" * 60)

    # -----------------------------------------------------
    # Step 1: Load embedding model
    # -----------------------------------------------------

    print("\n[1/5] Loading embedding model...")

    model = load_embedding_model()


    # -----------------------------------------------------
    # Step 2: Connect to ChromaDB
    # -----------------------------------------------------

    print("\n[2/5] Connecting to evidence database...")

    collection = get_collection()


    # -----------------------------------------------------
    # Step 3: Generate query embedding
    # -----------------------------------------------------

    print("\n[3/5] Generating query embedding...")

    query_embedding = embed_query(
        model,
        question
    )


    # -----------------------------------------------------
    # Step 4: Retrieve relevant evidence
    # -----------------------------------------------------

    print("\n[4/5] Retrieving relevant video evidence...")

    raw_results = retrieve_evidence(
        collection,
        query_embedding,
        top_k=top_k
    )

    retrieved_results = format_results(
        raw_results
    )

    print(
        f"\nRetrieved "
        f"{len(retrieved_results)} "
        f"evidence chunks."
    )


    # -----------------------------------------------------
    # Step 5: Build prompt
    # -----------------------------------------------------

    print("\n[5/5] Building evidence-based prompt...")

    prompt = build_prompt(
        question,
        retrieved_results
    )


    # -----------------------------------------------------
    # Generate LLM response
    # -----------------------------------------------------

    print("\nSending evidence to LLM...")

    answer = generate_response(
        prompt
    )


    return answer


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print(
        "\nMultimodal Deepfake "
        "Evidence Assistant"
    )

    print(
        "\nAsk a question about "
        "the analyzed video."
    )

    question = input(
        "\n> "
    )


    # -----------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------

    answer = run_rag_pipeline(
        question,
        top_k=3
    )


    # -----------------------------------------------------
    # Display answer
    # -----------------------------------------------------

    display_response(
        answer
    )