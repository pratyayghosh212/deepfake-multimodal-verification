from sentence_transformers import SentenceTransformer
import chromadb


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_PATH = "data/chroma"

COLLECTION_NAME = "deepfake_evidence"


# =========================================================
# Initialize embedding model
# =========================================================

def load_embedding_model():
    """
    Load the same embedding model used when
    generating the evidence embeddings.
    """

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    return model


# =========================================================
# Connect to ChromaDB
# =========================================================

def get_collection():

    print("Connecting to Chroma database...")

    print(f"Database path: {CHROMA_PATH}")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collections = client.list_collections()

    print("\nAvailable collections:")

    if not collections:
        print("No collections found.")

    else:
        for item in collections:
            print(f"- {item.name}")

    print()

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Collection loaded: {COLLECTION_NAME}"
    )

    print(
        f"Documents available: {collection.count()}"
    )

    return collection
# =========================================================
# Create query embedding
# =========================================================

def embed_query(model, query):
    """
    Convert the user's question into a vector embedding.
    """

    embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    return embedding.tolist()


# =========================================================
# Retrieve relevant evidence
# =========================================================

def retrieve_evidence(
    collection,
    query_embedding,
    top_k=3
):
    """
    Search ChromaDB for the evidence chunks
    most relevant to the user's query.
    """

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


# =========================================================
# Format retrieved results
# =========================================================

def format_results(results):
    """
    Convert Chroma results into a cleaner
    Python structure.
    """

    formatted_results = []

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    for index in range(len(documents)):

        formatted_results.append(
            {
                "rank": index + 1,
                "document": documents[index],
                "metadata": metadatas[index],
                "distance": distances[index]
            }
        )

    return formatted_results


# =========================================================
# Display results
# =========================================================

def print_results(results):
    """
    Print retrieved evidence in a readable format.
    """

    print("\n" + "=" * 60)

    print("RETRIEVED EVIDENCE")

    print("=" * 60)

    for item in results:

        print(
            f"\nRank: {item['rank']}"
        )

        print(
            f"Distance: {item['distance']:.4f}"
        )

        print(
            f"Metadata: {item['metadata']}"
        )

        print("\nEvidence:")

        print(
            item["document"]
        )

        print(
            "\n" + "-" * 60
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_embedding_model()

    # -----------------------------------------------------
    # Connect to Chroma
    # -----------------------------------------------------

    collection = get_collection()

    # -----------------------------------------------------
    # Ask question
    # -----------------------------------------------------

    query = input(
        "\nAsk a question about the video evidence:\n> "
    )

    # -----------------------------------------------------
    # Create query embedding
    # -----------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = embed_query(
        model,
        query
    )

    # -----------------------------------------------------
    # Retrieve evidence
    # -----------------------------------------------------

    print(
        "Searching for relevant evidence..."
    )

    raw_results = retrieve_evidence(
        collection,
        query_embedding,
        top_k=3
    )

    # -----------------------------------------------------
    # Format results
    # -----------------------------------------------------

    results = format_results(
        raw_results
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print_results(
        results
    )