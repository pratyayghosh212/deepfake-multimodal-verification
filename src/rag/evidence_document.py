import json
from pathlib import Path


def load_evidence(path):
    """Load unified evidence."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def visual_to_text(item):
    """Convert visual evidence into searchable text."""

    timestamp = item["timestamp"]
    face_count = item["face_count"]

    if face_count == 0:
        return (
            f"Visual evidence at {timestamp:.1f} seconds. "
            "No face was detected in the frame."
        )

    descriptions = []

    for index, face in enumerate(item["faces"]):

        descriptions.append(
            f"Face {index + 1}: "
            f"confidence {face['confidence']:.4f}, "
            f"bounding box x={face['x']}, "
            f"y={face['y']}, "
            f"width={face['width']}, "
            f"height={face['height']}."
        )

    return (
        f"Visual evidence at {timestamp:.1f} seconds. "
        f"{face_count} face(s) detected. "
        + " ".join(descriptions)
    )


def speech_to_text(item):
    """Convert speech evidence into searchable text."""

    return (
        f"Speech from {item['start']:.1f} "
        f"to {item['end']:.1f} seconds. "
        f"Language: {item['language']}. "
        f"Transcript: {item['text']}"
    )


def create_documents(evidence):
    """
    Convert structured evidence into
    searchable text documents.
    """

    documents = []

    for item in evidence:

        if item["type"] == "speech":

            text = speech_to_text(item)

            metadata = {
                "evidence_id": item["id"],
                "type": "speech",
                "start": item["start"],
                "end": item["end"],
                "source": item["source"],
                "language": item["language"]
            }

        elif item["type"] == "visual":

            text = visual_to_text(item)

            metadata = {
                "evidence_id": item["id"],
                "type": "visual",
                "timestamp": item["timestamp"],
                "frame": item["frame"],
                "source": item["source"]
            }

        else:
            continue

        documents.append({
            "id": item["id"],
            "text": text,
            "metadata": metadata
        })

    return documents


def save_documents(documents, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":

    evidence_path = (
        "data/evidence/evidence.json"
    )

    output_path = (
        "data/evidence/documents.json"
    )

    print("Loading evidence...")

    evidence = load_evidence(
        evidence_path
    )

    print("Creating searchable documents...")

    documents = create_documents(
        evidence
    )

    save_documents(
        documents,
        output_path
    )

    print(
        f"Created {len(documents)} documents."
    )

    print(
        f"Saved to: {output_path}"
    )