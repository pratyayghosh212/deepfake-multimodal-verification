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
# Helper
# =========================================================

def clamp(
    value,
    minimum=0.0,
    maximum=1.0
):
    """
    Restrict a value to [0, 1].
    """

    return max(
        minimum,
        min(float(value), maximum)
    )


# =========================================================
# Temporal anomaly normalization
# =========================================================

def normalize_deformation(value):
    """
    Convert landmark deformation into a
    supporting anomaly score.

    IMPORTANT:

    This is NOT a deepfake probability.

    Facial landmark movement can occur because
    of natural head movement, expression changes,
    camera motion, or tracking instability.

    Therefore this is used only as supporting
    evidence.
    """

    if value is None:
        return 0.0

    value = float(value)

    # Conservative threshold.
    #
    # A deformation around 0.20 is treated as
    # high temporal irregularity.

    threshold = 0.20

    score = value / threshold

    return clamp(score)


# =========================================================
# Audio signal
# =========================================================

def get_audio_signal(audio_item):
    """
    Extract audio information.

    IMPORTANT:

    RMS, ZCR, and spectral centroid are acoustic
    features, NOT deepfake probabilities.

    Therefore ordinary acoustic properties are
    preserved as evidence but are NOT directly
    converted into a fake probability.
    """

    if not audio_item:

        return {

            "available": False,

            "rms": 0.0,

            "zero_crossing_rate": 0.0,

            "spectral_centroid": 0.0,

            "spectral_bandwidth": 0.0,

            "spectral_rolloff": 0.0,

            "anomaly_score": None
        }

    features = audio_item.get(
        "features",
        {}
    ) or {}

    rms = float(
        features.get(
            "rms_energy",
            0.0
        ) or 0.0
    )

    zcr = float(
        features.get(
            "zero_crossing_rate",
            0.0
        ) or 0.0
    )

    centroid = float(
        features.get(
            "spectral_centroid",
            0.0
        ) or 0.0
    )

    bandwidth = float(
        features.get(
            "spectral_bandwidth",
            0.0
        ) or 0.0
    )

    rolloff = float(
        features.get(
            "spectral_rolloff",
            0.0
        ) or 0.0
    )

    # Audio is considered measurable when RMS
    # is above a very small threshold.

    available = rms >= 0.0001

    return {

        "available": available,

        "rms": round(
            rms,
            6
        ),

        "zero_crossing_rate": round(
            zcr,
            6
        ),

        "spectral_centroid": round(
            centroid,
            3
        ),

        "spectral_bandwidth": round(
            bandwidth,
            3
        ),

        "spectral_rolloff": round(
            rolloff,
            3
        ),

        # We do NOT invent a deepfake score
        # from ordinary acoustic features.

        "anomaly_score": None
    }


# =========================================================
# Lip-sync validation
# =========================================================

def get_lipsync_signal(
    lipsync_item,
    audio_available
):
    """
    Validate lip-sync evidence.

    Lip-sync influences fusion only when:

        1. Lip-sync data exists
        2. Audio is available
        3. Synchronization ratio is valid

    Otherwise it is marked unavailable.
    """

    if not lipsync_item:

        return {

            "available": False,

            "synchronization_ratio": None,

            "anomaly_score": None
        }

    if not audio_available:

        return {

            "available": False,

            "synchronization_ratio": None,

            "anomaly_score": None
        }

    synchronization_ratio = lipsync_item.get(
        "synchronization_ratio"
    )

    if synchronization_ratio is None:

        return {

            "available": False,

            "synchronization_ratio": None,

            "anomaly_score": None
        }

    try:

        synchronization_ratio = float(
            synchronization_ratio
        )

    except (
        ValueError,
        TypeError
    ):

        return {

            "available": False,

            "synchronization_ratio": None,

            "anomaly_score": None
        }

    # Reject invalid ratios.

    if (
        synchronization_ratio < 0.0
        or synchronization_ratio > 1.0
    ):

        return {

            "available": False,

            "synchronization_ratio": None,

            "anomaly_score": None
        }

    anomaly_score = (
        1.0 -
        synchronization_ratio
    )

    return {

        "available": True,

        "synchronization_ratio": round(
            synchronization_ratio,
            6
        ),

        "anomaly_score": round(
            anomaly_score,
            6
        )
    }


# =========================================================
# Timestamp lookup
# =========================================================

def find_temporal_by_filename(
    filename,
    temporal_lookup
):
    """
    Find temporal analysis using filename.
    """

    return temporal_lookup.get(
        filename,
        {}
    )


def find_audio_at_timestamp(
    timestamp,
    audio_data
):
    """
    Find audio window containing timestamp.
    """

    for audio in audio_data:

        start = float(
            audio.get(
                "start",
                0.0
            ) or 0.0
        )

        end = float(
            audio.get(
                "end",
                0.0
            ) or 0.0
        )

        if start <= timestamp < end:

            return audio

    return None


def find_lipsync_by_timestamp(
    timestamp,
    lipsync_data
):
    """
    Find lip-sync evidence corresponding
    to a visual frame timestamp.
    """

    # -------------------------------------------------
    # Handle dictionary-based JSON formats
    # -------------------------------------------------

    if isinstance(
        lipsync_data,
        dict
    ):

        for key in [

            "results",

            "frames",

            "entries",

            "analysis",

            "lipsync",

            "data"
        ]:

            if key in lipsync_data:

                lipsync_data = lipsync_data[key]

                break

        else:

            print(
                "WARNING: Unsupported lip-sync "
                "dictionary structure."
            )

            return None

    # -------------------------------------------------
    # Validate list
    # -------------------------------------------------

    if not isinstance(
        lipsync_data,
        list
    ):

        return None

    # -------------------------------------------------
    # Search entries
    # -------------------------------------------------

    for item in lipsync_data:

        if not isinstance(
            item,
            dict
        ):

            continue

        # -------------------------------------------------
        # Exact timestamp
        # -------------------------------------------------

        item_timestamp = item.get(
            "timestamp"
        )

        if item_timestamp is not None:

            try:

                if abs(
                    float(item_timestamp)
                    - float(timestamp)
                ) < 0.11:

                    return item

            except (
                ValueError,
                TypeError
            ):

                continue

        # -------------------------------------------------
        # Time range
        # -------------------------------------------------

        start = item.get(
            "start"
        )

        end = item.get(
            "end"
        )

        if (
            start is not None
            and end is not None
        ):

            try:

                if (
                    float(start)
                    <= float(timestamp)
                    <= float(end)
                ):

                    return item

            except (
                ValueError,
                TypeError
            ):

                continue

    return None


# =========================================================
# Fusion
# =========================================================

def fuse_visual_evidence(
    visual_data,
    temporal_data,
    audio_data,
    lipsync_data
):
    """
    Fuse multimodal evidence frame-by-frame.

    Signals preserved:

        1. Visual deepfake detector
        2. Face detection information
        3. Temporal landmark analysis
        4. Audio acoustic features
        5. Lip-sync evidence
        6. Aligned speech / transcript

    Fusion strategy:

        Visual detector:
            Primary learned signal

        Temporal analysis:
            Supporting heuristic signal

        Lip-sync:
            Used only when valid

        Audio:
            Used as contextual evidence only

        Transcript:
            Preserved for RAG retrieval but does
            NOT directly affect fake probability.
    """

    fused_frames = []

    # -----------------------------------------------------
    # Create temporal lookup
    # -----------------------------------------------------

    temporal_lookup = {

        item.get("filename"): item

        for item in temporal_data

        if item.get("filename") is not None
    }

    # -----------------------------------------------------
    # Process each visual frame
    # -----------------------------------------------------

    for frame in visual_data:

        timestamp = float(
            frame.get(
                "timestamp",
                0.0
            ) or 0.0
        )

        filename = frame.get(
            "frame",
            "unknown"
        )

        # =================================================
        # Preserve face detection information
        # =================================================

        face_count = int(
            frame.get(
                "face_count",
                0
            ) or 0
        )

        faces = frame.get(
            "faces",
            []
        ) or []

        # =================================================
        # Preserve aligned speech / transcript
        # =================================================

        aligned_speech = frame.get(
            "aligned_speech",
            []
        ) or []

        # =================================================
        # Visual detector
        # =================================================

        deepfake_model = frame.get(
            "deepfake_model",
            {}
        ) or {}

        fake_probability = float(
            deepfake_model.get(
                "fake_probability",
                0.0
            ) or 0.0
        )

        real_probability = float(
            deepfake_model.get(
                "real_probability",
                0.0
            ) or 0.0
        )

        predicted_label = deepfake_model.get(
            "predicted_label",
            "unknown"
        )

        fake_probability = clamp(
            fake_probability
        )

        real_probability = clamp(
            real_probability
        )

        # =================================================
        # Temporal analysis
        # =================================================

        temporal_item = (
            find_temporal_by_filename(
                filename,
                temporal_lookup
            )
        )

        frame_temporal = frame.get(
            "temporal",
            {}
        ) or {}

        mean_deformation = float(
            temporal_item.get(
                "mean_deformation",
                frame_temporal.get(
                    "mean_deformation",
                    0.0
                )
            ) or 0.0
        )

        std_deformation = float(
            temporal_item.get(
                "std_deformation",
                frame_temporal.get(
                    "std_deformation",
                    0.0
                )
            ) or 0.0
        )

        max_deformation = float(
            temporal_item.get(
                "max_deformation",
                frame_temporal.get(
                    "max_deformation",
                    0.0
                )
            ) or 0.0
        )

        temporal_anomaly = (
            normalize_deformation(
                mean_deformation
            )
        )

        # =================================================
        # Audio
        # =================================================

        audio_item = (
            find_audio_at_timestamp(
                timestamp,
                audio_data
            )
        )

        audio_signal = (
            get_audio_signal(
                audio_item
            )
        )

        # =================================================
        # Lip-sync
        # =================================================

        lipsync_item = (
            find_lipsync_by_timestamp(
                timestamp,
                lipsync_data
            )
        )

        lipsync_signal = (
            get_lipsync_signal(

                lipsync_item,

                audio_signal[
                    "available"
                ]
            )
        )

        # =================================================
        # Weighted fusion
        # =================================================

        if lipsync_signal["available"]:

            fusion_score = (

                0.70 *
                fake_probability

                +

                0.15 *
                temporal_anomaly

                +

                0.15 *
                lipsync_signal[
                    "anomaly_score"
                ]
            )

            active_modalities = [

                "visual",

                "temporal",

                "lip_sync"
            ]

        else:

            # Renormalized weights when
            # lip-sync is unavailable.

            fusion_score = (

                0.80 *
                fake_probability

                +

                0.20 *
                temporal_anomaly
            )

            active_modalities = [

                "visual",

                "temporal"
            ]

        # Audio and speech are preserved as
        # contextual evidence but do not directly
        # influence the suspicion score.

        if audio_signal["available"]:

            active_modalities.append(
                "audio"
            )

        if aligned_speech:

            active_modalities.append(
                "speech"
            )

        fusion_score = clamp(
            fusion_score
        )

        # =================================================
        # Evidence level
        # =================================================

        if fusion_score >= 0.70:

            evidence_level = "high"

        elif fusion_score >= 0.45:

            evidence_level = "medium"

        else:

            evidence_level = "low"

        # =================================================
        # Create fused frame
        # =================================================

        fused_frame = {

            "timestamp": round(
                timestamp,
                3
            ),

            "frame": filename,

            # ---------------------------------------------
            # Face detection evidence
            # ---------------------------------------------

            "face_detection": {

                "face_count": face_count,

                "faces": faces
            },

            # ---------------------------------------------
            # Speech / transcript evidence
            # ---------------------------------------------

            "speech": {

                "segments": aligned_speech
            },

            # ---------------------------------------------
            # Multimodal signals
            # ---------------------------------------------

            "signals": {

                "visual": {

                    "predicted_label":
                        predicted_label,

                    "real_probability":
                        round(
                            real_probability,
                            6
                        ),

                    "fake_probability":
                        round(
                            fake_probability,
                            6
                        )
                },

                "temporal": {

                    "mean_deformation":
                        round(
                            mean_deformation,
                            6
                        ),

                    "std_deformation":
                        round(
                            std_deformation,
                            6
                        ),

                    "max_deformation":
                        round(
                            max_deformation,
                            6
                        ),

                    "anomaly_score":
                        round(
                            temporal_anomaly,
                            6
                        )
                },

                "audio":
                    audio_signal,

                "lip_sync":
                    lipsync_signal
            },

            # ---------------------------------------------
            # Fusion result
            # ---------------------------------------------

            "fusion": {

                "score":
                    round(
                        fusion_score,
                        6
                    ),

                "evidence_level":
                    evidence_level,

                "active_modalities":
                    active_modalities
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
    Calculate overall video-level
    suspicion score.

    We use:

        80% average frame score
        20% maximum frame score

    This prevents one unusual frame from
    dominating the entire video classification.
    """

    if not fused_frames:

        return {

            "overall_score": 0.0,

            "maximum_frame_score": 0.0,

            "average_frame_score": 0.0,

            "assessment":
                "insufficient_evidence"
        }

    scores = [

        frame["fusion"]["score"]

        for frame in fused_frames
    ]

    average_score = (
        sum(scores)
        /
        len(scores)
    )

    maximum_score = max(
        scores
    )

    overall_score = (

        0.80 *
        average_score

        +

        0.20 *
        maximum_score
    )

    overall_score = clamp(
        overall_score
    )

    # -----------------------------------------------------
    # Assessment
    # -----------------------------------------------------

    if overall_score >= 0.70:

        assessment = "high_suspicion"

    elif overall_score >= 0.45:

        assessment = "moderate_suspicion"

    else:

        assessment = "low_suspicion"

    return {

        "overall_score":
            round(
                overall_score,
                6
            ),

        "maximum_frame_score":
            round(
                maximum_score,
                6
            ),

        "average_frame_score":
            round(
                average_score,
                6
            ),

        "assessment":
            assessment
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
    # Start
    # =====================================================

    print()

    print("=" * 60)
    print("MULTIMODAL EVIDENCE FUSION")
    print("=" * 60)

    print()

    # =====================================================
    # Load unified evidence
    # =====================================================

    print(
        "Loading unified evidence..."
    )

    all_evidence = load_json(
        evidence_path
    )

    # -----------------------------------------------------
    # Extract multimodal visual frames
    # -----------------------------------------------------

    visual_data = [

        item

        for item in all_evidence

        if item.get("type")
        == "multimodal_visual"
    ]

    # -----------------------------------------------------
    # Count speech evidence
    # -----------------------------------------------------

    speech_evidence_count = len(

        [

            item

            for item in all_evidence

            if item.get("type")
            == "speech"
        ]
    )

    print(
        f"Multimodal visual frames: "
        f"{len(visual_data)}"
    )

    print(
        f"Speech evidence segments: "
        f"{speech_evidence_count}"
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

    lipsync_file = Path(
        lipsync_path
    )

    if lipsync_file.exists():

        lipsync_data = load_json(
            lipsync_path
        )

        if isinstance(
            lipsync_data,
            list
        ):

            lipsync_count = len(
                lipsync_data
            )

        elif isinstance(
            lipsync_data,
            dict
        ):

            lipsync_count = "dictionary format"

        else:

            lipsync_count = 0

        print(
            f"Lip-sync entries: "
            f"{lipsync_count}"
        )

    else:

        print(
            "Lip-sync file not found."
        )

        print(
            "Continuing without lip-sync evidence."
        )

        lipsync_data = []

    # =====================================================
    # Validate
    # =====================================================

    if not visual_data:

        raise ValueError(

            "No multimodal visual evidence found. "
            "Check evidence.json and ensure the type "
            "is 'multimodal_visual'."
        )

    # =====================================================
    # Fuse
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
    # Overall score
    # =====================================================

    overall = calculate_overall_score(
        fused_frames
    )

    # =====================================================
    # Count preserved modalities
    # =====================================================

    frames_with_speech = sum(

        1

        for frame in fused_frames

        if frame.get(
            "speech",
            {}
        ).get(
            "segments",
            []
        )
    )

    frames_with_audio = sum(

        1

        for frame in fused_frames

        if frame.get(
            "signals",
            {}
        ).get(
            "audio",
            {}
        ).get(
            "available",
            False
        )
    )

    frames_with_lipsync = sum(

        1

        for frame in fused_frames

        if frame.get(
            "signals",
            {}
        ).get(
            "lip_sync",
            {}
        ).get(
            "available",
            False
        )
    )

    # =====================================================
    # Final result
    # =====================================================

    result = {

        "system": {

            "name":
                "Multimodal Deepfake Evidence Fusion",

            "version":
                "2.1"
        },

        "overall":
            overall,

        "metadata": {

            "frames_fused":
                len(fused_frames),

            "speech_evidence_segments":
                speech_evidence_count,

            "frames_with_aligned_speech":
                frames_with_speech,

            "frames_with_audio":
                frames_with_audio,

            "frames_with_lipsync":
                frames_with_lipsync
        },

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
    # Summary
    # =====================================================

    print()

    print("-" * 60)

    print(
        "Multimodal fusion completed."
    )

    print(
        f"Frames fused: "
        f"{len(fused_frames)}"
    )

    print(
        f"Frames with aligned speech: "
        f"{frames_with_speech}"
    )

    print(
        f"Frames with usable audio: "
        f"{frames_with_audio}"
    )

    print(
        f"Frames with lip-sync evidence: "
        f"{frames_with_lipsync}"
    )

    print()

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

    print()

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print("-" * 60)