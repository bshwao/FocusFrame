from main.main import yolo, model, preprocess
import cv2
import torch
import numpy as np
import pytest

# Define the prompts for the scenarios we want to test
prompts = [
    "A person holding a drink",
    "A person on their phone",
    "A person looking at their watch",
]

# Define device for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to import and tokenize with CLIP if available; otherwise continue without tokenized text
try:
    import clip
    text = clip.tokenize(prompts).to(device)
except Exception:
    # CLIP not available or tokenization failed; proceed without `text`
    text = None

# Function to simulate detection


def simulate_detection(frame):
    results = yolo(frame, device=device, verbose=False)
    return results

# Test cases for detection scenarios


@pytest.mark.parametrize("prompt_index", range(len(prompts)))
def test_detection_scenarios(prompt_index):
    # Load a test image (replace with actual test images)
    test_image_path = f"tests/test_images/test_image_{prompt_index}.jpg"
    frame = cv2.imread(test_image_path)

    if frame is None:
        pytest.fail(f"Could not load image at {test_image_path}")

    # Simulate detection
    results = simulate_detection(frame)

    assert results is not None, "Detection results should not be None"

    # Check if the expected prompt is detected
    detected = False
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) == prompt_index:  # Assuming class indices match prompt indices
                detected = True
                break

    assert detected, f"Expected prompt '{prompts[prompt_index]}' was not detected."
