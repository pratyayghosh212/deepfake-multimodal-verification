import json
from pathlib import Path

import torch

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


class VisualDeepfakeDetector:
    """
    Pretrained visual deepfake detector.

    Responsibilities:
    1. Load the pretrained model once.
    2. Predict individual images.
    3. Analyze all extracted video frames.
    4. Preserve the original frame timestamps.
    5. Save structured visual evidence.
    """

    def __init__(
        self,
        model_name="yithh/ViT-DeepfakeDetection"
    ):

        self.model_name = model_name

        # -------------------------------------------------
        # Select device
        # -------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        # -------------------------------------------------
        # Load image processor
        # -------------------------------------------------

        print(
            "Loading image processor..."
        )

        self.processor = (
            AutoImageProcessor.from_pretrained(
                self.model_name
            )
        )

        # -------------------------------------------------
        # Load pretrained model
        # -------------------------------------------------

        print(
            "Loading visual deepfake model..."
        )

        self.model = (
            AutoModelForImageClassification
            .from_pretrained(
                self.model_name
            )
        )

        # Move model to CPU/GPU
        self.model.to(
            self.device
        )

        # Inference mode
        self.model.eval()

        # -------------------------------------------------
        # Display model labels
        # -------------------------------------------------

        print(
            "Model labels:"
        )

        print(
            self.model.config.id2label
        )

        print(
            "Model loaded successfully."
        )

    # =====================================================
    # Predict a single image
    # =====================================================

    def predict(
        self,
        image_path
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Load image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Preprocess
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # Move tensors to device
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Inference
        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        # Raw logits
        logits = outputs.logits

        # Convert to probabilities
        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # Predicted class
        predicted_class = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        # Predicted label
        label = self.model.config.id2label[
            predicted_class
        ]

        # Confidence
        confidence = probabilities[
            0,
            predicted_class
        ].item()

        # All class probabilities
        class_probabilities = {}

        for class_id, probability in enumerate(
            probabilities[0]
        ):

            class_label = (
                self.model.config.id2label[
                    class_id
                ]
            )

            class_probabilities[
                class_label
            ] = round(
                probability.item(),
                6
            )

        return {
            "image": image_path.name,
            "label": label,
            "confidence": round(
                confidence,
                6
            ),
            "probabilities": class_probabilities
        }

    # =====================================================
    # Analyze all video frames
    # =====================================================

    def analyze_video_frames(
        self,
        frames_dir,
        metadata_path,
        output_path,
        batch_size=8
    ):
        """
        Run visual deepfake detection on all
        extracted video frames.

        The timestamps come from frames.json,
        rather than being calculated from the
        frame number.
        """

        frames_dir = Path(
            frames_dir
        )

        metadata_path = Path(
            metadata_path
        )

        output_path = Path(
            output_path
        )

        # -------------------------------------------------
        # Check directories/files
        # -------------------------------------------------

        if not frames_dir.exists():

            raise FileNotFoundError(
                f"Frames directory not found: "
                f"{frames_dir}"
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{metadata_path}"
            )

        # -------------------------------------------------
        # Load timestamp metadata
        # -------------------------------------------------

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        print(
            f"Loaded timestamp metadata "
            f"for {len(metadata)} frames."
        )

        # -------------------------------------------------
        # Prepare frame information
        # -------------------------------------------------

        frame_items = []

        for item in metadata:

            frame_path = (
                frames_dir /
                item["filename"]
            )

            if not frame_path.exists():

                print(
                    f"Warning: missing frame "
                    f"{frame_path}"
                )

                continue

            frame_items.append({
                "path": frame_path,
                "filename": item["filename"],
                "timestamp": item["timestamp"]
            })

        print(
            f"Found {len(frame_items)} frames "
            f"for visual analysis."
        )

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        results = []

        # -------------------------------------------------
        # Process frames in batches
        # -------------------------------------------------

        for start in range(
            0,
            len(frame_items),
            batch_size
        ):

            batch = frame_items[
                start:start + batch_size
            ]

            images = []

            valid_items = []

            # ---------------------------------------------
            # Load images
            # ---------------------------------------------

            for item in batch:

                try:

                    image = Image.open(
                        item["path"]
                    ).convert("RGB")

                    images.append(
                        image
                    )

                    valid_items.append(
                        item
                    )

                except Exception as error:

                    print(
                        f"Could not read "
                        f"{item['filename']}: "
                        f"{error}"
                    )

            if not images:
                continue

            # ---------------------------------------------
            # Preprocess entire batch
            # ---------------------------------------------

            inputs = self.processor(
                images=images,
                return_tensors="pt"
            )

            # Move tensors to device
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # ---------------------------------------------
            # Model inference
            # ---------------------------------------------

            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

            # ---------------------------------------------
            # Convert logits → probabilities
            # ---------------------------------------------

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            # ---------------------------------------------
            # Process each prediction
            # ---------------------------------------------

            for index, item in enumerate(
                valid_items
            ):

                frame_probabilities = (
                    probabilities[index]
                )

                predicted_class = torch.argmax(
                    frame_probabilities
                ).item()

                predicted_label = (
                    self.model.config.id2label[
                        predicted_class
                    ]
                )

                # Get real probability
                real_probability = 0.0

                # Get fake probability
                fake_probability = 0.0

                for class_id, probability in enumerate(
                    frame_probabilities
                ):

                    label = (
                        self.model.config.id2label[
                            class_id
                        ].lower()
                    )

                    if label == "real":

                        real_probability = (
                            probability.item()
                        )

                    elif label == "fake":

                        fake_probability = (
                            probability.item()
                        )

                result = {
                    "timestamp": item["timestamp"],
                    "filename": item["filename"],
                    "predicted_label": predicted_label,
                    "real_probability": round(
                        real_probability,
                        6
                    ),
                    "fake_probability": round(
                        fake_probability,
                        6
                    )
                }

                results.append(
                    result
                )

                print(
                    f"{item['filename']} | "
                    f"{item['timestamp']:.3f}s | "
                    f"{predicted_label} | "
                    f"real: {real_probability:.4f} | "
                    f"fake: {fake_probability:.4f}"
                )

        # -------------------------------------------------
        # Sort by timestamp
        # -------------------------------------------------

        results.sort(
            key=lambda item: item["timestamp"]
        )

        # -------------------------------------------------
        # Save results
        # -------------------------------------------------

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
                results,
                file,
                indent=4
            )

        print()
        print(
            f"Saved visual model analysis to:"
        )
        print(output_path)

        return results


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    detector = VisualDeepfakeDetector()

    detector.analyze_video_frames(
        frames_dir="data/frames",
        metadata_path="data/frames/frames.json",
        output_path=(
            "data/visual_analysis/"
            "visual_model_analysis.json"
        ),
        batch_size=8
    )