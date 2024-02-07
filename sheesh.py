from ultralytics import YOLO
import cv2
import torch
import os
import multiprocessing

def load_yolo_model(model_name, weight_path):
    # Load the YOLO model
    yolo_model = YOLO(model_name).cuda()

    # Check if the weight file exists
    assert os.path.isfile(weight_path), f"File not found: {weight_path}"

    # Load the state dictionary
    state_dict = torch.load(weight_path)

    # Filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in yolo_model.state_dict()}

    # Load the filtered state dictionary
    yolo_model.load_state_dict(state_dict, strict=False)

    # Return the loaded model
    return yolo_model

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows only

    # Load the YOLO model using the function
    yolo_model = load_yolo_model("yolov8n.yaml", 'C:\\Users\\jdich\\thesisproject\\runs\\detect\\train3\\weights\\best.pt')

    yolo_model.eval()

    # Use a context manager to handle the video capture object
    with cv2.VideoCapture(0) as cap:
        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()

            # Perform inference using YOLO
            results = yolo_model(frame)

            # Draw bounding boxes on the frame
            boxes = results.xyxy[0][:, :4].cpu().numpy()
            confidences = results.xyxy[0][:, 4].cpu().numpy()
            indices = (confidences > 0.5).nonzero()[0]  # Keep only detections with confidence > 0.5

            for i in indices:
                x_min, y_min, x_max, y_max = boxes[i]
                class_id = int(results.xyxy[0][i, 5].cpu().numpy())

                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {confidences[i]:.2f}", (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('YOLO Real-time Detection', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # No need to release the video capture object and close the OpenCV windows
        # The context manager will do it automatically