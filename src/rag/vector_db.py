import json
from pathlib import Path

import chromadb


# =========================================================
# Configuration
# =========================================================

CHROMA_PATH = "data/chroma"

COLLECTION_NAME = "deepfake_evidence"

CHUNKS_PATH = (
    "data/evidence/chunks.json"
)

EMBEDDINGS_PATH = (
    "data/evidence/embeddings.json"
)


# =========================================================
# JSON utilities
# =========================================================

def load_json(path):
    """Load JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# Load chunks
# =========================================================

def load_chunks(path):
    """Load temporal chunks."""

    chunks = load_json(path)

    if not isinstance(chunks, list):
        raise ValueError(
            "chunks.json must contain a list."
        )

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    return chunks


# =========================================================
# Load embeddings
# =========================================================

def load_embeddings(path):
    """
    Load embeddings.

    Expected format:

    [
        {
            "id": "chunk_0000",
            "text": "...",
            "embedding": [384 floats]
        }
    ]
    """

    embeddings = load_json(path)

    if not isinstance(embeddings, list):
        raise ValueError(
            "embeddings.json must contain a list."
        )

    print(
        f"Embeddings loaded: "
        f"{len(embeddings)}"
    )

    return embeddings


# =========================================================
# Validate data
# =========================================================

def validate_data(
    chunks,
    embeddings
):
    """Validate chunk/embedding consistency."""

    if len(chunks) != len(embeddings):

        raise ValueError(
            "Number of chunks and embeddings "
            "does not match."
        )

    for index, item in enumerate(
        embeddings
    ):

        if "embedding" not in item:

            raise ValueError(
                f"Embedding missing at index "
                f"{index}."
            )

        vector = item["embedding"]

        if not isinstance(
            vector,
            list
        ):

            raise ValueError(
                f"Embedding at index {index} "
                "is not a list."
            )

        if len(vector) == 0:

            raise ValueError(
                f"Embedding at index {index} "
                "is empty."
            )

    print(
        "Data validation passed."
    )


# =========================================================
# Prepare Chroma records
# =========================================================

def prepare_records(
    chunks,
    embeddings
):
    """
    Convert our JSON representation
    into Chroma-compatible records.
    """

    ids = []

    documents = []

    metadatas = []

    embedding_vectors = []

    for index, chunk in enumerate(
        chunks
    ):

        embedding_item = embeddings[index]

        # ---------------------------------------------
        # ID
        # ---------------------------------------------

        chunk_id = chunk.get(
            "id",
            f"chunk_{index:04d}"
        )

        # ---------------------------------------------
        # Text
        # ---------------------------------------------

        text = chunk.get(
            "text",
            ""
        )

        # ---------------------------------------------
        # Embedding
        # ---------------------------------------------

        vector = embedding_item[
            "embedding"
        ]

        vector = [
            float(value)
            for value in vector
        ]

        # ---------------------------------------------
        # Metadata
        # ---------------------------------------------

        metadata = {
            "chunk_id": str(chunk_id),
            "start": float(
                chunk.get(
                    "start",
                    0.0
                )
            ),
            "end": float(
                chunk.get(
                    "end",
                    0.0
                )
            ),
            "frame_count": len(
                chunk.get(
                    "frames",
                    []
                )
            )
        }

        ids.append(
            str(chunk_id)
        )

        documents.append(
            str(text)
        )

        metadatas.append(
            metadata
        )

        embedding_vectors.append(
            vector
        )

    return (
        ids,
        documents,
        metadatas,
        embedding_vectors
    )


# =========================================================
# Initialize Chroma
# =========================================================

def initialize_chroma():

    print(
        "Initializing Chroma database..."
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
                "Multimodal deepfake evidence"
        }
    )

    print(
        f"Collection ready: "
        f"{COLLECTION_NAME}"
    )

    return client, collection


# =========================================================
# Store embeddings
# =========================================================

def store_embeddings(
    collection,
    chunks,
    embeddings
):

    (
        ids,
        documents,
        metadatas,
        embedding_vectors
    ) = prepare_records(
        chunks,
        embeddings
    )

    print(
        f"Storing "
        f"{len(embedding_vectors)} "
        f"embeddings..."
    )

    # ---------------------------------------------
    # Sanity check
    # ---------------------------------------------

    if len(
        embedding_vectors
    ) > 0:

        print(
            "Embedding dimension: "
            f"{len(embedding_vectors[0])}"
        )

    # ---------------------------------------------
    # Chroma upsert
    # ---------------------------------------------

    collection.upsert(

        ids=ids,

        embeddings=embedding_vectors,

        documents=documents,

        metadatas=metadatas
    )

    print(
        "Embeddings stored successfully."
    )


# =========================================================
# Verify collection
# =========================================================

def verify_collection(
    collection
):

    print()
    print(
        "Verifying Chroma collection..."
    )

    count = collection.count()

    print(
        f"Documents stored: {count}"
    )

    if count > 0:

        result = collection.get(
            limit=min(
                3,
                count
            ),
            include=[
                "documents",
                "metadatas"
            ]
        )

        print()

        print(
            "Sample stored documents:"
        )

        for index, document in enumerate(
            result["documents"]
        ):

            print(
                f"\n[{index}] "
                f"{document[:200]}..."
            )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print(
        "Loading chunks..."
    )

    chunks = load_chunks(
        CHUNKS_PATH
    )

    print(
        "Loading embeddings..."
    )

    embeddings = load_embeddings(
        EMBEDDINGS_PATH
    )

    # ---------------------------------------------
    # Validate
    # ---------------------------------------------

    validate_data(
        chunks,
        embeddings
    )

    # ---------------------------------------------
    # Initialize database
    # ---------------------------------------------

    client, collection = (
        initialize_chroma()
    )

    # ---------------------------------------------
    # Store vectors
    # ---------------------------------------------

    store_embeddings(
        collection,
        chunks,
        embeddings
    )

    # ---------------------------------------------
    # Verify
    # ---------------------------------------------

    verify_collection(
        collection
    )

    print()
    print(
        "--------------------------------"
    )

    print(
        "Vector database creation completed."
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Documents: {collection.count()}"
    )

    print(
        f"Database: {CHROMA_PATH}"
    )

    print(
        "--------------------------------"
    )