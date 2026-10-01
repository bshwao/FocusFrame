from pathlib import Path
import json
import cv2

# Define the output directory for annotations
output_dir = Path("../data/annotations")
output_dir.mkdir(parents=True, exist_ok=True)

# Define the scenarios for annotation collection
scenarios = [
    "A person holding a drink",
    "A person on their phone",
    "A person looking at their watch",
]

# Initialize a dictionary to hold annotations
annotations = {}

# Function to collect annotations from the webcam


def collect_annotations():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("Press 'q' to quit the annotation collection.")
    print("Press 's' to save the current frame with the selected scenario.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break

        cv2.imshow("Annotation Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            scenario_index = int(
                input(f"Select scenario index (0-{len(scenarios)-1}): "))
            if 0 <= scenario_index < len(scenarios):
                scenario = scenarios[scenario_index]
                frame_name = f"frame_{len(annotations)}.jpg"
                cv2.imwrite(str(output_dir / frame_name), frame)
                annotations[frame_name] = scenario
                print(f"Saved {frame_name} with scenario: {scenario}")
            else:
                print("Invalid scenario index.")

    cap.release()
    cv2.destroyAllWindows()

    # Save annotations to a JSON file
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    collect_annotations()
