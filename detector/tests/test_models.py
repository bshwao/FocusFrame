import pytest
from main.yolo import YOLO
from main.clip_model import CLIPModel


def test_yolo_model_loading():
    model_name = "yolov8n.pt"
    yolo_model = YOLO(model_name)
    assert yolo_model is not None, "YOLO model failed to load"


def test_clip_model_loading():
    clip_model = CLIPModel()
    assert clip_model is not None, "CLIP model failed to load"


def test_yolo_inference():
    model_name = "yolov8n.pt"
    yolo_model = YOLO(model_name)
    test_image = "path/to/test/image.jpg"  # Replace with a valid image path
    results = yolo_model(test_image)
    assert results is not None, "YOLO inference failed"
    assert len(results) > 0, "No detections made by YOLO"


def test_clip_inference():
    clip_model = CLIPModel()
    test_image = "path/to/test/image.jpg"  # Replace with a valid image path
    prompts = ["A person holding a drink",
               "A person on their phone", "A person looking at their watch"]
    results = clip_model.infer(test_image, prompts)
    assert results is not None, "CLIP inference failed"
    assert len(results) == len(
        prompts), "CLIP inference did not return expected number of results"
