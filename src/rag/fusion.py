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


def save_json(data, output_path):
    """
    Save data as formatted JSON.
    """

    output_path = Path(output_path)

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
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# Basic helper
# =========================================================

def clamp(
    value,
    minimum=0.0,
    maximum=1.0
):
    """
    Restrict a value to the range [0, 1].
    """

    return max(
        minimum,
        min(value, maximum)
    )


# =========================================================
# Temporal anomaly score
# =========================================================

def normalize_deformation(value):
    """
    Convert landmark deformation into a
    normalized temporal anomaly score.

    The landmark analyzer produces deformation
    values rather than probabilities.

    Therefore this function simply maps the
    deformation into the range [0, 1].

    Example:

        0.01 -> small anomaly
        0.10 -> moderate anomaly
        0.20+ -> high anomaly
    """

    threshold = 0.20

    score = (
        value /
        threshold
    )

    return clamp(
        score
    )


# =========================================================
# Audio anomaly score
# =========================================================

def get_audio_anomaly_score(
    audio_item
):
    """
    Calculate a simple heuristic audio
    anomaly score.

    Current audio analysis provides:

        RMS
        ZCR
        spectral centroid

    These are NOT deepfake probabilities.

    They are only supporting audio signals.
    """

    if not audio_item:

        return 0.0

    # -----------------------------------------------------
    # Read RMS
    # -----------------------------------------------------

    rms = audio_item.get(
        "rms",
        0.0
    )

    # -----------------------------------------------------
    # Read zero crossing rate
    #
    # Support both possible field names.
    # -----------------------------------------------------

    zcr = audio_item.get(
        "zero_crossing_rate",
        audio_item.get(
            "zcr",
            0.0
        )
    )

    # -----------------------------------------------------
    # Read spectral centroid
    #
    # Support both possible field names.
    # -----------------------------------------------------

    centroid = audio_item.get(
        "spectral_centroid",
        audio_item.get(
            "centroid",
            0.0
        )
    )

    # -----------------------------------------------------
    # Normalize individual signals
    # -----------------------------------------------------

    rms_score = clamp(
        rms / 0.10
    )

    zcr_score = clamp(
        zcr / 0.40
    )

    centroid_score = clamp(
        centroid / 4000.0
    )

    # -----------------------------------------------------
    # Weighted audio score
    # -----------------------------------------------------

    score = (
        0.40 * rms_score +
        0.30 * zcr_score +
        0.30 * centroid_score
    )

    return clamp(
        score
    )


# =========================================================
# Lip-sync score
# =========================================================

def get_lipsync_score(
    lipsync_data
):
    """
    Convert synchronization ratio into
    a synchronization anomaly score.

    High synchronization is good.

    Therefore:

        anomaly = 1 - synchronization_ratio
    """

    synchronization_ratio = lipsync_data.get(
        "synchronization_ratio",
        0.0
    )

    synchronization_ratio = clamp(
        synchronization_ratio
    )

    anomaly_score = (
        1.0 -
        synchronization_ratio
    )

    return round(
        anomaly_score,
        6
    )


# =========================================================
# Frame-level multimodal fusion
# =========================================================

def fuse_visual_evidence(
    visual_data,
    temporal_data,
    audio_data,
    lipsync_data
):
    """
    Combine visual, temporal, audio and
    lip-sync signals for every frame.

    Current signals:

        1. ViT fake probability
        2. Landmark temporal deformation
        3. Audio anomaly
        4. Lip-sync anomaly

    Returns frame-level fused evidence.
    """

    # =====================================================
    # Temporal lookup
    # =====================================================

    temporal_lookup = {
        item.get("filename"): item
        for item in temporal_data
    }

    # =====================================================
    # Audio lookup
    #
    # Audio analysis is performed in 0.2 second
    # windows. We match the window midpoint
    # with the visual frame timestamp.
    # =====================================================

    audio_lookup = {}

    for item in audio_data:

        start = item.get(
            "start",
            0.0
        )

        end = item.get(
            "end",
            start
        )

        midpoint = (
            start +
            end
        ) / 2.0

        audio_lookup[
            round(
                midpoint,
                1
            )
        ] = item

    # =====================================================
    # Global lip-sync anomaly
    # =====================================================

    lipsync_anomaly = get_lipsync_score(
        lipsync_data
    )

    # =====================================================
    # Store results
    # =====================================================

    fused_frames = []

    # =====================================================
    # Process every visual frame
    # =====================================================

    for frame in visual_data:

        # -------------------------------------------------
        # Get frame information
        # -------------------------------------------------

        filename = frame.get(
            "frame",
            frame.get(
                "filename"
            )
        )

        timestamp = frame.get(
            "timestamp",
            0.0
        )

        # -------------------------------------------------
        # ViT deepfake signal
        # -------------------------------------------------

        deepfake_model = frame.get(
            "deepfake_model",
            {}
        )

        fake_probability = deepfake_model.get(
            "fake_probability",
            0.0
        )

        fake_probability = clamp(
            fake_probability
        )

        # -------------------------------------------------
        # Temporal landmark signal
        # -------------------------------------------------

        temporal = temporal_lookup.get(
            filename,
            {}
        )

        mean_deformation = temporal.get(
            "mean_deformation",
            0.0
        )

        temporal_score = (
            normalize_deformation(
                mean_deformation
            )
        )

        # -------------------------------------------------
        # Audio signal
        # -------------------------------------------------

        audio_key = round(
            timestamp,
            1
        )

        audio_item = audio_lookup.get(
            audio_key,
            {}
        )

        audio_score = (
            get_audio_anomaly_score(
                audio_item
            )
        )

        # -------------------------------------------------
        # Lip-sync signal
        #
        # Current lip-sync analyzer provides
        # a global synchronization ratio.
        #
        # Therefore the same score is temporarily
        # used for every frame.
        # -------------------------------------------------

        frame_lipsync_score = (
            lipsync_anomaly
        )

        # =================================================
        # Multimodal fusion
        # =================================================
        #
        # Current weights:
        #
        # ViT             = 50%
        # Temporal        = 20%
        # Audio           = 10%
        # Lip-sync        = 20%
        #
        # These are heuristic weights.
        # They are NOT learned weights.
        # =================================================

        fused_score = (

            0.50 *
            fake_probability

            +

            0.20 *
            temporal_score

            +

            0.10 *
            audio_score

            +

            0.20 *
            frame_lipsync_score
        )

        fused_score = clamp(
            fused_score
        )

        # =================================================
        # Evidence level
        # =================================================

        if fused_score >= 0.70:

            evidence_level = "high"

        elif fused_score >= 0.40:

            evidence_level = "medium"

        else:

            evidence_level = "low"

        # =================================================
        # Create fused frame
        # =================================================

        fused_frame = {

            "timestamp": timestamp,

            "frame": filename,

            "signals": {

                # -----------------------------------------
                # Visual model
                # -----------------------------------------

                "visual": {

                    "fake_probability": round(
                        fake_probability,
                        6
                    )
                },

                # -----------------------------------------
                # Temporal landmarks
                # -----------------------------------------

                "temporal": {

                    "mean_deformation": round(
                        mean_deformation,
                        6
                    ),

                    "anomaly_score": round(
                        temporal_score,
                        6
                    )
                },

                # -----------------------------------------
                # Audio
                # -----------------------------------------

                "audio": {

                    "anomaly_score": round(
                        audio_score,
                        6
                    )
                },

                # -----------------------------------------
                # Lip-sync
                # -----------------------------------------

                "lip_sync": {

                    "synchronization_ratio": round(
                        1.0 -
                        frame_lipsync_score,
                        6
                    ),

                    "anomaly_score": round(
                        frame_lipsync_score,
                        6
                    )
                }
            },

            # ---------------------------------------------
            # Final frame-level fusion
            # ---------------------------------------------

            "fusion": {

                "score": round(
                    fused_score,
                    6
                ),

                "evidence_level":
                    evidence_level
            }
        }

        fused_frames.append(
            fused_frame
        )

    return fused_frames


# =========================================================
# Overall video score
# =========================================================

def calculate_overall_score(
    fused_frames
):
    """
    Calculate the overall video-level
    suspicion score.

    We combine:

        70% average frame score
        30% maximum frame score

    This means one unusual frame does not
    automatically classify the entire video
    as fake.
    """

    if not fused_frames:

        return {

            "overall_score": 0.0,

            "maximum_frame_score": 0.0,

            "average_frame_score": 0.0,

            "assessment":
                "insufficient_evidence"
        }

    # -----------------------------------------------------
    # Extract frame scores
    # -----------------------------------------------------

    scores = [

        frame["fusion"]["score"]

        for frame in fused_frames
    ]

    # -----------------------------------------------------
    # Average score
    # -----------------------------------------------------

    average_score = (
        sum(scores) /
        len(scores)
    )

    # -----------------------------------------------------
    # Maximum score
    # -----------------------------------------------------

    maximum_score = max(
        scores
    )

    # -----------------------------------------------------
    # Overall score
    # -----------------------------------------------------

    overall_score = (

        0.70 *
        average_score

        +

        0.30 *
        maximum_score
    )

    overall_score = clamp(
        overall_score
    )

    # -----------------------------------------------------
    # Overall assessment
    # -----------------------------------------------------

    if overall_score >= 0.70:

        assessment = (
            "high_suspicion"
        )

    elif overall_score >= 0.40:

        assessment = (
            "moderate_suspicion"
        )

    else:

        assessment = (
            "low_suspicion"
        )

    return {

        "overall_score": round(
            overall_score,
            6
        ),

        "maximum_frame_score": round(
            maximum_score,
            6
        ),

        "average_frame_score": round(
            average_score,
            6
        ),

        "assessment": assessment
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # Input paths
    # =====================================================

    evidence_path = (
        "data/evidence/"
        "evidence.json"
    )

    temporal_path = (
        "data/visual_analysis/"
        "normalized_landmark_analysis.json"
    )

    audio_path = (
        "data/audio/"
        "audio_analysis.json"
    )

    lipsync_path = (
        "data/visual_analysis/"
        "lipsync_analysis.json"
    )

    # =====================================================
    # Output path
    # =====================================================

    output_path = (
        "data/evidence/"
        "fused_evidence.json"
    )

    # =====================================================
    # Load visual evidence
    # =====================================================

    print(
        "Loading visual evidence..."
    )

    all_evidence = load_json(
        evidence_path
    )

    # evidence.json contains both
    # speech and visual evidence.
    #
    # We only want visual frames here.

    visual_data = [

        item

        for item in all_evidence

        if item.get("type") == "visual"
    ]

    print(
        f"Visual frames: "
        f"{len(visual_data)}"
    )

    # =====================================================
    # Load temporal analysis
    # =====================================================

    print(
        "Loading temporal analysis..."
    )

    temporal_data = load_json(
        temporal_path
    )

    print(
        f"Temporal frames: "
        f"{len(temporal_data)}"
    )

    # =====================================================
    # Load audio analysis
    # =====================================================

    print(
        "Loading audio analysis..."
    )

    audio_data = load_json(
        audio_path
    )

    print(
        f"Audio windows: "
        f"{len(audio_data)}"
    )

    # =====================================================
    # Load lip-sync analysis
    # =====================================================

    print(
        "Loading lip-sync analysis..."
    )

    lipsync_data = load_json(
        lipsync_path
    )

    # =====================================================
    # Fuse all signals
    # =====================================================

    print()
    print(
        "Fusing multimodal evidence..."
    )

    fused_frames = fuse_visual_evidence(

        visual_data,

        temporal_data,

        audio_data,

        lipsync_data
    )

    # =====================================================
    # Calculate overall score
    # =====================================================

    overall = calculate_overall_score(
        fused_frames
    )

    # =====================================================
    # Final result
    # =====================================================

    result = {

        "system": {

            "name":
                "Multimodal Deepfake Evidence Fusion",

            "version":
                "1.0"
        },

        "overall":
            overall,

        "frames":
            fused_frames
    }

    # =====================================================
    # Save
    # =====================================================

    save_json(
        result,
        output_path
    )

    # =====================================================
    # Print summary
    # =====================================================

    print()
    print(
        "--------------------------------"
    )

    print(
        "Multimodal fusion completed."
    )

    print(
        f"Frames fused: "
        f"{len(fused_frames)}"
    )

    print(
        f"Overall score: "
        f"{overall['overall_score']:.4f}"
    )

    print(
        f"Assessment: "
        f"{overall['assessment']}"
    )

    print(
        f"Maximum frame score: "
        f"{overall['maximum_frame_score']:.4f}"
    )

    print(
        f"Average frame score: "
        f"{overall['average_frame_score']:.4f}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print(
        "--------------------------------"
    )