import os
import json
import cv2
import numpy as np


def export_dataset(raw_data_dir, annotations_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(
        raw_data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    annotations = []

    for image_file in image_files:
        image_path = os.path.join(raw_data_dir, image_file)
        annotation_file = os.path.join(annotations_dir, image_file.replace(
            '.jpg', '.json').replace('.png', '.json'))

        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
                annotations.append({
                    'image': image_file,
                    'annotations': annotation
                })

    output_file = os.path.join(output_dir, 'dataset.json')
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)

    print(f"Dataset exported to {output_file}")


if __name__ == "__main__":
    raw_data_directory = '../data/raw'
    annotations_directory = '../data/annotations'
    output_directory = '../data/processed'

    export_dataset(raw_data_directory, annotations_directory, output_directory)
