import json
from pathlib import Path


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


def save_json(data, file_path):
    """
    Save JSON data to a file.
    """

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
# Safe conversion helpers
# =========================================================

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.

    Handles:
        None
        missing values
        invalid strings
    """

    if value is None:
        return default

    try:

        return float(value)

    except (
        ValueError,
        TypeError
    ):

        return default


# =========================================================
# Frame description
# =========================================================

def frame_to_text(frame):
    """
    Convert one fused frame into a searchable
    natural-language evidence description.

    The generated text is used for:

        - chunking
        - embeddings
        - Chroma retrieval
        - RAG evidence

    IMPORTANT:

    This function preserves both:

        - technical deepfake evidence
        - semantic evidence such as transcript
          and face information

    This allows the RAG system to answer both:

        "Is this video suspicious?"

    and:

        "What is the audio transcript?"
        "What happens in the video?"
    """

    # =====================================================
    # Basic frame information
    # =====================================================

    timestamp = safe_float(
        frame.get(
            "timestamp",
            0.0
        )
    )

    filename = frame.get(
        "frame",
        "unknown"
    )

    signals = frame.get(
        "signals",
        {}
    ) or {}

    fusion = frame.get(
        "fusion",
        {}
    ) or {}

    # =====================================================
    # Face detection information
    # =====================================================

    face_detection = frame.get(
        "face_detection",
        {}
    ) or {}

    face_count = int(
        face_detection.get(
            "face_count",
            0
        ) or 0
    )

    faces = face_detection.get(
        "faces",
        []
    ) or []

    # =====================================================
    # Speech / transcript information
    # =====================================================

    speech = frame.get(
        "speech",
        {}
    ) or {}

    speech_segments = speech.get(
        "segments",
        []
    ) or []

    transcript_parts = []

    for segment in speech_segments:

        if not isinstance(
            segment,
            dict
        ):
            continue

        # Try common transcript keys

        segment_text = (

            segment.get("text")

            or segment.get("transcript")

            or segment.get("speech")

            or ""
        )

        if segment_text:

            transcript_parts.append(
                str(segment_text).strip()
            )

    transcript_text = " ".join(
        transcript_parts
    )

    # =====================================================
    # Visual signal
    # =====================================================

    visual = signals.get(
        "visual",
        {}
    ) or {}

    predicted_label = visual.get(
        "predicted_label",
        "unknown"
    )

    fake_probability = safe_float(
        visual.get(
            "fake_probability",
            0.0
        )
    )

    real_probability = safe_float(
        visual.get(
            "real_probability",
            0.0
        )
    )

    # =====================================================
    # Temporal signal
    # =====================================================

    temporal = signals.get(
        "temporal",
        {}
    ) or {}

    mean_deformation = safe_float(
        temporal.get(
            "mean_deformation",
            0.0
        )
    )

    std_deformation = safe_float(
        temporal.get(
            "std_deformation",
            0.0
        )
    )

    max_deformation = safe_float(
        temporal.get(
            "max_deformation",
            0.0
        )
    )

    temporal_anomaly = safe_float(
        temporal.get(
            "anomaly_score",
            0.0
        )
    )

    # =====================================================
    # Audio signal
    # =====================================================

    audio = signals.get(
        "audio",
        {}
    ) or {}

    audio_available = bool(
        audio.get(
            "available",
            False
        )
    )

    rms = safe_float(
        audio.get(
            "rms",
            0.0
        )
    )

    zcr = safe_float(
        audio.get(
            "zero_crossing_rate",
            0.0
        )
    )

    centroid = safe_float(
        audio.get(
            "spectral_centroid",
            0.0
        )
    )

    # =====================================================
    # Lip-sync signal
    # =====================================================

    lip_sync = signals.get(
        "lip_sync",
        {}
    ) or {}

    lipsync_available = bool(
        lip_sync.get(
            "available",
            False
        )
    )

    synchronization_ratio = (
        lip_sync.get(
            "synchronization_ratio"
        )
    )

    lipsync_anomaly = (
        lip_sync.get(
            "anomaly_score"
        )
    )

    # =====================================================
    # Fusion signal
    # =====================================================

    fusion_score = safe_float(
        fusion.get(
            "score",
            0.0
        )
    )

    evidence_level = fusion.get(
        "evidence_level",
        "unknown"
    )

    active_modalities = fusion.get(
        "active_modalities",
        []
    ) or []

    # =====================================================
    # Face description
    # =====================================================

    if face_count == 0:

        face_text = (
            "No face was detected in this frame. "
        )

    elif face_count == 1:

        face_text = (
            "One face was detected in this frame. "
        )

    else:

        face_text = (
            f"{face_count} faces were detected "
            f"in this frame. "
        )

    # =====================================================
    # Transcript description
    # =====================================================

    if transcript_text:

        speech_text = (
            f"The aligned audio transcript at this "
            f"timestamp is: '{transcript_text}'. "
        )

    else:

        speech_text = (
            "No transcript segment was aligned "
            "with this frame. "
        )

    # =====================================================
    # Audio description
    # =====================================================

    if audio_available:

        audio_text = (
            f"Usable audio was available. "
            f"Audio analysis reported RMS "
            f"{rms:.4f}, "
            f"zero-crossing rate "
            f"{zcr:.4f}, "
            f"and spectral centroid "
            f"{centroid:.2f} Hz. "
        )

    else:

        audio_text = (
            "No reliable audio evidence was "
            "available for this timestamp. "
        )

    # =====================================================
    # Lip-sync description
    # =====================================================

    if (
        lipsync_available
        and synchronization_ratio is not None
    ):

        synchronization_ratio = safe_float(
            synchronization_ratio
        )

        lipsync_text = (
            f"Lip-sync synchronization ratio was "
            f"{synchronization_ratio:.4f}. "
        )

        if lipsync_anomaly is not None:

            lipsync_anomaly = safe_float(
                lipsync_anomaly
            )

            lipsync_text += (
                f"Lip-sync anomaly score was "
                f"{lipsync_anomaly:.4f}. "
            )

    else:

        lipsync_text = (
            "No reliable lip-sync measurement was "
            "available for this timestamp. "
        )

    # =====================================================
    # Active modalities
    # =====================================================

    if active_modalities:

        modalities_text = ", ".join(
            active_modalities
        )

    else:

        modalities_text = "unknown"

    # =====================================================
    # Build searchable evidence text
    # =====================================================

    text = (

        f"VIDEO FRAME EVIDENCE. "

        f"Timestamp: {timestamp:.2f} seconds. "

        f"Frame: {filename}. "

        f"{face_text}"

        f"{speech_text}"

        f"The visual deepfake detector predicted "
        f"'{predicted_label}' with fake probability "
        f"{fake_probability:.4f} and real probability "
        f"{real_probability:.4f}. "

        f"Facial temporal analysis reported mean "
        f"deformation {mean_deformation:.4f}, "
        f"standard deviation {std_deformation:.4f}, "
        f"maximum deformation {max_deformation:.4f}, "
        f"and temporal anomaly score "
        f"{temporal_anomaly:.4f}. "

        f"{audio_text}"

        f"{lipsync_text}"

        f"The multimodal fusion score was "
        f"{fusion_score:.4f}, "
        f"with evidence level "
        f"'{evidence_level}'. "

        f"Active modalities were: "
        f"{modalities_text}."
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

        Chunk 0:
            0.0 - 1.0 seconds

        Chunk 1:
            1.0 - 2.0 seconds

        Chunk 2:
            2.0 - 3.0 seconds

    Each chunk contains:

        - chunk ID
        - start/end time
        - frame IDs
        - number of frames
        - aggregated statistics
        - evidence levels
        - searchable natural-language text
    """

    if not frames:

        return []

    # =====================================================
    # Sort frames chronologically
    # =====================================================

    frames = sorted(

        frames,

        key=lambda item: safe_float(
            item.get(
                "timestamp",
                0.0
            )
        )
    )

    chunks = []

    # =====================================================
    # Determine total duration
    # =====================================================

    max_timestamp = max(

        safe_float(
            frame.get(
                "timestamp",
                0.0
            )
        )

        for frame in frames
    )

    # Frames are extracted approximately every 0.2 seconds.
    # Add a small interval so the final frame belongs
    # to a temporal range.

    total_duration = (
        max_timestamp + 0.2
    )

    # =====================================================
    # Calculate number of chunks
    # =====================================================

    number_of_chunks = int(
        total_duration // chunk_duration
    )

    if (
        total_duration % chunk_duration
        != 0
    ):

        number_of_chunks += 1

    # =====================================================
    # Create temporal chunks
    # =====================================================

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
        # Select frames belonging to this chunk
        # -------------------------------------------------

        chunk_frames = [

            frame

            for frame in frames

            if (

                safe_float(
                    frame.get(
                        "timestamp",
                        0.0
                    )
                )
                >= start

                and

                safe_float(
                    frame.get(
                        "timestamp",
                        0.0
                    )
                )
                < end
            )
        ]

        # Skip empty intervals.

        if not chunk_frames:

            continue

        # =================================================
        # Generate searchable text
        # =================================================

        frame_texts = [

            frame_to_text(frame)

            for frame in chunk_frames
        ]

        combined_text = " ".join(
            frame_texts
        )

        # =================================================
        # Frame IDs
        # =================================================

        frame_ids = [

            frame.get(
                "frame"
            )

            for frame in chunk_frames

            if frame.get(
                "frame"
            ) is not None
        ]

        # =================================================
        # Fusion statistics
        # =================================================

        fusion_scores = [

            safe_float(

                frame.get(
                    "fusion",
                    {}
                ).get(
                    "score",
                    0.0
                )
            )

            for frame in chunk_frames
        ]

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

        # =================================================
        # Visual fake probability statistics
        # =================================================

        fake_probabilities = [

            safe_float(

                frame.get(
                    "signals",
                    {}
                ).get(
                    "visual",
                    {}
                ).get(
                    "fake_probability",
                    0.0
                )
            )

            for frame in chunk_frames
        ]

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

        # =================================================
        # Temporal anomaly statistics
        # =================================================

        temporal_anomalies = [

            safe_float(

                frame.get(
                    "signals",
                    {}
                ).get(
                    "temporal",
                    {}
                ).get(
                    "anomaly_score",
                    0.0
                )
            )

            for frame in chunk_frames
        ]

        average_temporal_anomaly = (

            sum(temporal_anomalies) /
            len(temporal_anomalies)

            if temporal_anomalies

            else 0.0
        )

        maximum_temporal_anomaly = (

            max(temporal_anomalies)

            if temporal_anomalies

            else 0.0
        )

        # =================================================
        # Evidence levels
        # =================================================

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

        # =================================================
        # Active modalities
        # =================================================

        active_modalities = set()

        for frame in chunk_frames:

            modalities = frame.get(
                "fusion",
                {}
            ).get(
                "active_modalities",
                []
            )

            if isinstance(
                modalities,
                list
            ):

                for modality in modalities:

                    active_modalities.add(
                        modality
                    )

        # =================================================
        # Create chunk
        # =================================================

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

            # Kept for compatibility with
            # other RAG components.

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
                ),

                "average_temporal_anomaly": round(
                    average_temporal_anomaly,
                    6
                ),

                "maximum_temporal_anomaly": round(
                    maximum_temporal_anomaly,
                    6
                )
            },

            "evidence_levels": evidence_levels,

            "active_modalities": sorted(
                active_modalities
            ),

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

    # =====================================================
    # Header
    # =====================================================

    print()
    print("=" * 60)
    print("MULTIMODAL EVIDENCE CHUNKER")
    print("=" * 60)
    print()

    # =====================================================
    # Load fused evidence
    # =====================================================

    print(
        "Loading fused evidence..."
    )

    fused_data = load_json(
        input_path
    )

    # =====================================================
    # Extract frames
    # =====================================================

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

    # =====================================================
    # Validate
    # =====================================================

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

        if not isinstance(
            frame,
            dict
        ):

            raise ValueError(
                f"Frame {index} is not a valid "
                f"dictionary."
            )

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

    # =====================================================
    # Create chunks
    # =====================================================

    print()
    print(
        "Creating temporal chunks..."
    )

    chunks = create_chunks(

        frames,

        chunk_duration=1.0
    )

    # =====================================================
    # Validate chunks
    # =====================================================

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )

    # =====================================================
    # Save chunks
    # =====================================================

    save_json(

        chunks,

        output_path
    )

    # =====================================================
    # Summary
    # =====================================================

    total_frames = sum(

        chunk[
            "frame_count"
        ]

        for chunk in chunks
    )

    print()
    print("-" * 60)

    print(
        "Chunking completed."
    )

    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )

    print(
        f"Frames covered: "
        f"{total_frames}"
    )

    print(
        f"Chunk duration: "
        f"1.0 second"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print("-" * 60)