import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# JSON utilities
# =========================================================

def load_json(file_path):
    """
    Load JSON data from a file.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# Embedding model
# =========================================================

def load_embedding_model():
    """
    Load the pretrained sentence-transformer model.

    We use a pretrained model because we do not need
    to train an embedding model for this project.
    """

    print(
        f"Loading embedding model: "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    return model


# =========================================================
# Generate embeddings
# =========================================================

def generate_embeddings(
    chunks,
    model
):
    """
    Generate an embedding vector for every chunk.

    Each chunk's searchable text is converted into
    a numerical vector.
    """

    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    results = []

    for index, chunk in enumerate(chunks):

        vector = vectors[index]

        results.append({
            "id": chunk["id"],

            "text": chunk["text"],

            "embedding": vector.tolist(),

            "metadata": {
                "start": chunk["start"],
                "end": chunk["end"],
                "evidence_ids": chunk[
                    "evidence_ids"
                ],
                "frames": chunk[
                    "frames"
                ]
            }
        })

    return results


# =========================================================
# Save embeddings
# =========================================================

def save_embeddings(
    embeddings,
    output_path
):
    """
    Save embeddings to JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            embeddings,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    input_path = (
        "data/evidence/"
        "chunks.json"
    )

    output_path = (
        "data/evidence/"
        "embeddings.json"
    )

    # -----------------------------------------------------
    # Load chunks
    # -----------------------------------------------------

    print(
        "Loading chunks..."
    )

    chunks = load_json(
        input_path
    )

    print(
        f"Chunks loaded: "
        f"{len(chunks)}"
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_embedding_model()

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    embeddings = generate_embeddings(
        chunks,
        model
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_embeddings(
        embeddings,
        output_path
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()

    print(
        "--------------------------------"
    )

    print(
        "Embedding generation completed."
    )

    print(
        f"Embeddings created: "
        f"{len(embeddings)}"
    )

    if embeddings:

        dimension = len(
            embeddings[0]["embedding"]
        )

        print(
            f"Embedding dimension: "
            f"{dimension}"
        )

        print(
            f"First embedding values: "
            f"{embeddings[0]['embedding'][:5]}"
        )

    print(
        f"Saved to: {output_path}"
    )

    print(
        "--------------------------------"
    )