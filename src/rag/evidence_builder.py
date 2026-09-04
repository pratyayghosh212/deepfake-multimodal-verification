import json
from pathlib import Path


# =========================================================
# JSON utilities
# =========================================================

def load_json(file_path):
    """Load a JSON file."""

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# Speech evidence
# =========================================================

def build_speech_evidence(transcript):
    """
    Convert Whisper transcript segments
    into standardized speech evidence objects.
    """

    evidence = []

    for segment in transcript["segments"]:

        evidence_item = {

            "id": (
                f"speech_"
                f"{segment['segment_id']:04d}"
            ),

            "type": "speech",

            "start": segment["start"],
            "end": segment["end"],

            "text": segment["text"],

            "source": "whisper_transcription",

            "language": transcript["language"],

            "language_probability": transcript[
                "language_probability"
            ]
        }

        evidence.append(
            evidence_item
        )

    return evidence


# =========================================================
# Timestamp alignment
# =========================================================

def find_speech_at_timestamp(
    timestamp,
    speech_evidence
):
    """
    Find speech segments overlapping
    a particular timestamp.
    """

    matched_speech = []

    for speech in speech_evidence:

        if (
            speech["start"]
            <= timestamp
            <= speech["end"]
        ):

            matched_speech.append({

                "id": speech["id"],

                "start": speech["start"],

                "end": speech["end"],

                "text": speech["text"],

                "language": speech["language"]
            })

    return matched_speech


def find_audio_at_timestamp(
    timestamp,
    audio_data
):
    """
    Find the audio analysis window that
    contains the given visual timestamp.
    """

    for audio in audio_data:

        start = audio.get("start", 0.0)
        end = audio.get("end", 0.0)

        if start <= timestamp < end:

            return audio

    return None


# =========================================================
# Audio quality check
# =========================================================

def evaluate_audio_availability(audio_item):
    """
    Determine whether the audio window
    contains meaningful signal.

    This prevents missing/silent audio from
    being interpreted as suspicious evidence.
    """

    if not audio_item:
        return {
            "available": False,
            "reason": "No aligned audio window found."
        }

    features = audio_item.get(
        "features",
        {}
    )

    rms = features.get(
        "rms_energy",
        0.0
    )

    # Very low RMS generally means silence
    # or practically unavailable audio.

    if rms < 0.0001:

        return {
            "available": False,
            "reason": (
                "Audio window contains "
                "almost no measurable signal."
            )
        }

    return {
        "available": True,
        "reason": "Audio signal available."
    }


# =========================================================
# Visual evidence
# =========================================================

def build_visual_evidence(
    face_data,
    temporal_data,
    model_data,
    speech_evidence,
    audio_data
):
    """
    Combine multimodal evidence.

    Sources:

        1. MediaPipe face detection
        2. Normalized landmark temporal analysis
        3. Pretrained ViT deepfake detector
        4. Whisper speech transcription
        5. Librosa acoustic analysis
    """

    evidence = []

    # -----------------------------------------------------
    # Lookup tables
    # -----------------------------------------------------

    temporal_lookup = {

        item["filename"]: item

        for item in temporal_data
    }

    model_lookup = {

        item["filename"]: item

        for item in model_data
    }

    # -----------------------------------------------------
    # Process frames
    # -----------------------------------------------------

    for index, frame in enumerate(
        face_data
    ):

        filename = frame["filename"]

        timestamp = frame["timestamp"]

        faces = frame["faces"]

        # -------------------------------------------------
        # Temporal analysis
        # -------------------------------------------------

        temporal = temporal_lookup.get(
            filename,
            {}
        )

        # -------------------------------------------------
        # Visual deepfake model
        # -------------------------------------------------

        model = model_lookup.get(
            filename,
            {}
        )

        # -------------------------------------------------
        # Speech alignment
        # -------------------------------------------------

        aligned_speech = (
            find_speech_at_timestamp(
                timestamp,
                speech_evidence
            )
        )

        # -------------------------------------------------
        # Audio alignment
        # -------------------------------------------------

        aligned_audio = (
            find_audio_at_timestamp(
                timestamp,
                audio_data
            )
        )

        audio_status = (
            evaluate_audio_availability(
                aligned_audio
            )
        )

        # -------------------------------------------------
        # Build audio evidence
        # -------------------------------------------------

        if aligned_audio:

            audio_features = (
                aligned_audio.get(
                    "features",
                    {}
                )
            )

            audio_information = {

                "available": audio_status[
                    "available"
                ],

                "status": audio_status[
                    "reason"
                ],

                "start": aligned_audio.get(
                    "start"
                ),

                "end": aligned_audio.get(
                    "end"
                ),

                "rms_energy": audio_features.get(
                    "rms_energy",
                    0.0
                ),

                "zero_crossing_rate": (
                    audio_features.get(
                        "zero_crossing_rate",
                        0.0
                    )
                ),

                "spectral_centroid": (
                    audio_features.get(
                        "spectral_centroid",
                        0.0
                    )
                ),

                "spectral_bandwidth": (
                    audio_features.get(
                        "spectral_bandwidth",
                        0.0
                    )
                ),

                "spectral_rolloff": (
                    audio_features.get(
                        "spectral_rolloff",
                        0.0
                    )
                ),

                "source": aligned_audio.get(
                    "source",
                    "librosa_acoustic_analysis"
                )
            }

        else:

            audio_information = {

                "available": False,

                "status": (
                    "No aligned audio data found."
                )
            }

        # -------------------------------------------------
        # Build unified evidence item
        # -------------------------------------------------

        evidence_item = {

            "id": (
                f"visual_"
                f"{index:04d}"
            ),

            "type": "multimodal_visual",

            "timestamp": timestamp,

            "frame": filename,

            "source": [

                "mediapipe_face_detection",

                "normalized_landmark_analysis",

                "vit_deepfake_detector",

                "librosa_acoustic_analysis"
            ],

            # ---------------------------------------------
            # Face information
            # ---------------------------------------------

            "face_count": len(faces),

            "faces": faces,

            # ---------------------------------------------
            # Temporal facial information
            # ---------------------------------------------

            "temporal": {

                "mean_deformation": (
                    temporal.get(
                        "mean_deformation",
                        0.0
                    )
                ),

                "std_deformation": (
                    temporal.get(
                        "std_deformation",
                        0.0
                    )
                ),

                "max_deformation": (
                    temporal.get(
                        "max_deformation",
                        0.0
                    )
                )
            },

            # ---------------------------------------------
            # Visual deepfake model
            # ---------------------------------------------

            "deepfake_model": {

                "predicted_label": (
                    model.get(
                        "predicted_label"
                    )
                ),

                "real_probability": (
                    model.get(
                        "real_probability",
                        0.0
                    )
                ),

                "fake_probability": (
                    model.get(
                        "fake_probability",
                        0.0
                    )
                )
            },

            # ---------------------------------------------
            # Audio information
            # ---------------------------------------------

            "audio": audio_information,

            # ---------------------------------------------
            # Speech alignment
            # ---------------------------------------------

            "aligned_speech": aligned_speech
        }

        evidence.append(
            evidence_item
        )

    return evidence


# =========================================================
# Save evidence
# =========================================================

def save_evidence(
    evidence,
    output_path
):
    """Save unified evidence to JSON."""

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
            evidence,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MULTIMODAL EVIDENCE BUILDER")
    print("=" * 60)
    print()

    # -----------------------------------------------------
    # Input files
    # -----------------------------------------------------

    transcript_path = (
        "data/transcripts/"
        "transcript.json"
    )

    faces_path = (
        "data/visual_analysis/"
        "faces.json"
    )

    temporal_path = (
        "data/visual_analysis/"
        "normalized_landmark_analysis.json"
    )

    model_path = (
        "data/visual_analysis/"
        "visual_model_analysis.json"
    )

    audio_path = (
        "data/audio/"
        "audio_analysis.json"
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_path = (
        "data/evidence/"
        "evidence.json"
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("Loading transcript...")

    transcript = load_json(
        transcript_path
    )

    print("Loading face analysis...")

    face_data = load_json(
        faces_path
    )

    print("Loading temporal analysis...")

    temporal_data = load_json(
        temporal_path
    )

    print("Loading visual model analysis...")

    model_data = load_json(
        model_path
    )

    print("Loading audio analysis...")

    audio_data = load_json(
        audio_path
    )

    # -----------------------------------------------------
    # Build speech evidence
    # -----------------------------------------------------

    print()
    print("Building speech evidence...")

    speech_evidence = (
        build_speech_evidence(
            transcript
        )
    )

    # -----------------------------------------------------
    # Build multimodal visual evidence
    # -----------------------------------------------------

    print(
        "Building multimodal visual evidence..."
    )

    visual_evidence = (
        build_visual_evidence(

            face_data,

            temporal_data,

            model_data,

            speech_evidence,

            audio_data
        )
    )

    # -----------------------------------------------------
    # Combine evidence
    # -----------------------------------------------------

    unified_evidence = (
        speech_evidence +
        visual_evidence
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    unified_evidence.sort(

        key=lambda item:

        item.get(
            "start",

            item.get(
                "timestamp",
                0
            )
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_evidence(
        unified_evidence,
        output_path
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("EVIDENCE SUMMARY")
    print("=" * 60)

    print(
        f"Total evidence items: "
        f"{len(unified_evidence)}"
    )

    print(
        f"Speech evidence: "
        f"{len(speech_evidence)}"
    )

    print(
        f"Multimodal visual evidence: "
        f"{len(visual_evidence)}"
    )

    audio_available_count = sum(

        1

        for item in visual_evidence

        if item["audio"].get(
            "available",
            False
        )
    )

    print(
        f"Frames with usable audio: "
        f"{audio_available_count}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print("=" * 60)