
import json
from pathlib import Path


# =========================================================
# JSON utilities
# =========================================================

def load_json(file_path):
    """Load JSON data from a file."""

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json(data, file_path):
    """Save JSON data to a file."""

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# Frame description
# =========================================================

def frame_to_text(frame):
    """
    Convert one fused frame into a searchable
    natural-language evidence description.
    """

    timestamp = frame.get(
        "timestamp",
        0.0
    )

    filename = frame.get(
        "frame",
        "unknown"
    )

    signals = frame.get(
        "signals",
        {}
    )

    fusion = frame.get(
        "fusion",
        {}
    )

    # -----------------------------------------------------
    # Visual signal
    # -----------------------------------------------------

    visual = signals.get(
        "visual",
        {}
    )

    fake_probability = visual.get(
        "fake_probability",
        0.0
    )

    # -----------------------------------------------------
    # Temporal signal
    # -----------------------------------------------------

    temporal = signals.get(
        "temporal",
        {}
    )

    mean_deformation = temporal.get(
        "mean_deformation",
        0.0
    )

    temporal_anomaly = temporal.get(
        "anomaly_score",
        0.0
    )

    # -----------------------------------------------------
    # Audio signal
    # -----------------------------------------------------

    audio = signals.get(
        "audio",
        {}
    )

    audio_anomaly = audio.get(
        "anomaly_score",
        0.0
    )

    # Some versions of the fusion output may contain
    # additional audio features.
    rms = audio.get(
        "rms",
        0.0
    )

    zcr = audio.get(
        "zero_crossing_rate",
        0.0
    )

    centroid = audio.get(
        "spectral_centroid",
        0.0
    )

    # -----------------------------------------------------
    # Lip-sync signal
    # -----------------------------------------------------

    lip_sync = signals.get(
        "lip_sync",
        {}
    )

    synchronization_ratio = lip_sync.get(
        "synchronization_ratio",
        0.0
    )

    lipsync_anomaly = lip_sync.get(
        "anomaly_score",
        0.0
    )

    # -----------------------------------------------------
    # Fusion signal
    # -----------------------------------------------------

    fusion_score = fusion.get(
        "score",
        0.0
    )

    evidence_level = fusion.get(
        "evidence_level",
        "unknown"
    )

    # -----------------------------------------------------
    # Build searchable text
    # -----------------------------------------------------

    text = (
        f"At {timestamp:.2f} seconds, "
        f"frame {filename} contains multimodal deepfake evidence. "

        f"The visual deepfake detector reported a "
        f"fake probability of {fake_probability:.4f}. "

        f"Facial temporal analysis reported a mean deformation "
        f"of {mean_deformation:.4f} and temporal anomaly score "
        f"of {temporal_anomaly:.4f}. "

        f"Audio analysis reported RMS {rms:.4f}, "
        f"zero-crossing rate {zcr:.4f}, "
        f"spectral centroid {centroid:.2f} Hz, "
        f"and audio anomaly score {audio_anomaly:.4f}. "

        f"Lip-sync analysis reported a synchronization ratio "
        f"of {synchronization_ratio:.4f} and lip-sync anomaly "
        f"score of {lipsync_anomaly:.4f}. "

        f"The multimodal fusion score was "
        f"{fusion_score:.4f}, "
        f"with evidence level {evidence_level}."
    )

    return text


# =========================================================
# Temporal chunk creation
# =========================================================

def create_chunks(
    frames,
    chunk_duration=1.0
):
    """
    Group fused frames into temporal chunks.

    Example:

        0.0 - 1.0 seconds
        1.0 - 2.0 seconds
        2.0 - 3.0 seconds

    Each chunk contains:
        - chunk ID
        - temporal range
        - frame IDs
        - frame filenames
        - searchable text
        - aggregated fusion statistics
    """

    if not frames:
        return []

    # -----------------------------------------------------
    # Sort frames chronologically
    # -----------------------------------------------------

    frames = sorted(
        frames,
        key=lambda item: item.get(
            "timestamp",
            0.0
        )
    )

    chunks = []

    # -----------------------------------------------------
    # Determine chunk range
    # -----------------------------------------------------

    max_timestamp = max(
        frame.get(
            "timestamp",
            0.0
        )
        for frame in frames
    )

    total_duration = (
        max_timestamp +
        0.2
    )

    number_of_chunks = int(
        total_duration // chunk_duration
    )

    if total_duration % chunk_duration != 0:
        number_of_chunks += 1

    # -----------------------------------------------------
    # Create chunks
    # -----------------------------------------------------

    for chunk_index in range(
        number_of_chunks
    ):

        start = (
            chunk_index *
            chunk_duration
        )

        end = (
            start +
            chunk_duration
        )

        # -------------------------------------------------
        # Frames belonging to this time interval
        # -------------------------------------------------

        chunk_frames = [
            frame
            for frame in frames
            if (
                frame.get(
                    "timestamp",
                    0.0
                ) >= start
                and
                frame.get(
                    "timestamp",
                    0.0
                ) < end
            )
        ]

        if not chunk_frames:
            continue

        # -------------------------------------------------
        # Generate searchable text
        # -------------------------------------------------

        frame_texts = []

        for frame in chunk_frames:

            frame_texts.append(
                frame_to_text(frame)
            )

        combined_text = " ".join(
            frame_texts
        )

        # -------------------------------------------------
        # Frame IDs
        # -------------------------------------------------

        frame_ids = []

        for frame in chunk_frames:

            frame_id = frame.get(
                "frame"
            )

            if frame_id is not None:
                frame_ids.append(
                    frame_id
                )

        # -------------------------------------------------
        # Fusion scores
        # -------------------------------------------------

        fusion_scores = []

        for frame in chunk_frames:

            score = frame.get(
                "fusion",
                {}
            ).get(
                "score",
                0.0
            )

            fusion_scores.append(
                float(score)
            )

        average_score = (
            sum(fusion_scores) /
            len(fusion_scores)
            if fusion_scores
            else 0.0
        )

        maximum_score = (
            max(fusion_scores)
            if fusion_scores
            else 0.0
        )

        # -------------------------------------------------
        # Fake probabilities
        # -------------------------------------------------

        fake_probabilities = []

        for frame in chunk_frames:

            probability = frame.get(
                "signals",
                {}
            ).get(
                "visual",
                {}
            ).get(
                "fake_probability",
                0.0
            )

            fake_probabilities.append(
                float(probability)
            )

        average_fake_probability = (
            sum(fake_probabilities) /
            len(fake_probabilities)
            if fake_probabilities
            else 0.0
        )

        maximum_fake_probability = (
            max(fake_probabilities)
            if fake_probabilities
            else 0.0
        )

        # -------------------------------------------------
        # Evidence level
        # -------------------------------------------------

        evidence_levels = [
            frame.get(
                "fusion",
                {}
            ).get(
                "evidence_level",
                "unknown"
            )
            for frame in chunk_frames
        ]

        # -------------------------------------------------
        # Create chunk
        # -------------------------------------------------

        chunk = {
            "id": (
                f"chunk_"
                f"{chunk_index:04d}"
            ),

            "start": round(
                start,
                3
            ),

            "end": round(
                min(
                    end,
                    total_duration
                ),
                3
            ),

            "frame_count": len(
                chunk_frames
            ),

            "frame_ids": frame_ids,

            "frames": frame_ids,

            "evidence_ids": frame_ids,

            "statistics": {
                "average_fusion_score": round(
                    average_score,
                    6
                ),

                "maximum_fusion_score": round(
                    maximum_score,
                    6
                ),

                "average_fake_probability": round(
                    average_fake_probability,
                    6
                ),

                "maximum_fake_probability": round(
                    maximum_fake_probability,
                    6
                )
            },

            "evidence_levels": evidence_levels,

            "text": combined_text
        }

        chunks.append(
            chunk
        )

    return chunks


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    input_path = (
        "data/evidence/"
        "fused_evidence.json"
    )

    output_path = (
        "data/evidence/"
        "chunks.json"
    )

    # -----------------------------------------------------
    # Load fused evidence
    # -----------------------------------------------------

    print(
        "Loading fused evidence..."
    )

    fused_data = load_json(
        input_path
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # fused_evidence.json is a dictionary:
    #
    # {
    #     "system": ...,
    #     "overall": ...,
    #     "frames": [...]
    # }
    #
    # We need the frames list.
    # -----------------------------------------------------

    if isinstance(
        fused_data,
        dict
    ):

        frames = fused_data.get(
            "frames",
            []
        )

    elif isinstance(
        fused_data,
        list
    ):

        # Backward compatibility
        frames = fused_data

    else:

        raise ValueError(
            "Unsupported fused evidence format."
        )

    print(
        f"Evidence frames: "
        f"{len(frames)}"
    )

    # -----------------------------------------------------
    # Validate frames
    # -----------------------------------------------------

    if not frames:

        raise ValueError(
            "No fused frames found."
        )

    required_fields = [
        "timestamp",
        "frame",
        "signals",
        "fusion"
    ]

    for index, frame in enumerate(
        frames
    ):

        missing = [
            field
            for field in required_fields
            if field not in frame
        ]

        if missing:

            raise ValueError(
                f"Frame {index} is missing "
                f"fields: {missing}"
            )

    # -----------------------------------------------------
    # Create chunks
    # -----------------------------------------------------

    print()
    print(
        "Creating temporal chunks..."
    )

    chunks = create_chunks(
        frames,
        chunk_duration=1.0
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_json(
        chunks,
        output_path
    )

    print()
    print(
        "--------------------------------"
    )

    print(
        "Chunking completed."
    )

    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )

    print(
        f"Frames covered: "
        f"{sum(chunk['frame_count'] for chunk in chunks)}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print(
        "--------------------------------"
    )

