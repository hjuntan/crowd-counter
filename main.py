import cv2
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import torch
import sys, os

# Set manual testing flag
is_manual_testing = True
    
# Set the confidence threshold for person detection
confidence_threshold = 0.5

is_time_lapse = False

# Step 1: Load Hugging Face DETR for object detection
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def detect_people_in_frame(image):
    # Use the provided image (PIL.Image object) directly
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract bounding boxes, class labels and confidence scores
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes
    pred_scores = logits.softmax(-1)[..., :-1].max(-1).values  # Exclude the background class
    
    # Person class id in COCO dataset is 1
    person_class_id = 1
    boxes = pred_boxes[(logits.argmax(-1) == person_class_id) & (pred_scores > confidence_threshold)]
    
    # Convert boxes into (x, y, w, h) format expected by OpenCV
    boxes = boxes.tolist()  # convert to list of bounding boxes
    return [(box[0], box[1], box[2] - box[0], box[3] - box[1]) for box in boxes]  # x, y, w, h

if not is_manual_testing:
    # Step 2: Track detected people across frames manually and print the total count
    def track_people_in_video(video_path, frame_interval=30):
        cap = cv2.VideoCapture(video_path)
        
        # Total count of people detected
        total_people_count = 0
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:  # Process every Nth frame
                # Convert frame to PIL Image for detection
                frame_pil = Image.fromarray(frame)
                people_boxes = detect_people_in_frame(frame_pil)

                print(f"Frame {frame_count/frame_interval}: {len(people_boxes)} people detected")

                # Add the count of people detected in this frame to the total count
                total_people_count += len(people_boxes)

            frame_count += 1

        cap.release()
        
        # Print the total number of people detected in the video
        print(f"Total people detected in the entire video: {total_people_count}")

    
else:
    def track_people_in_video_manual():

        from scipy.io import loadmat

        file_path = './videos/mall_dataset/mall_gt.mat'
        data = loadmat(file_path)

        array_data = data['count']

        image_files = [f for f in os.listdir("videos/mall_dataset/frames") if f.endswith('.jpg')]
        
        # Total count of people detected
        total_people_count = 0

        # Absolute difference between the number of people detected and the actual count
        absolte_difference = 0

        for i in range(len(image_files)):
            image_path = os.path.join("videos/mall_dataset/frames", image_files[i])
            
            # Open the image using PIL
            image = Image.open(image_path)
            
            # Detect people in the image
            people_boxes = detect_people_in_frame(image)
            
            # Print the number of people detected in this image
            print(f"{image_files[i]}: Detected {len(people_boxes)} people")
            
            # Add the count of people detected in this image to the total count
            total_people_count += len(people_boxes)

            # Calculate the absolute difference between the number of people detected and the actual count
            absolte_difference += abs(array_data[i][0] - len(people_boxes))
        
        absolte_difference = absolte_difference / len(image_files)
        
        # Print the total number of people detected in all images
        print(f"Total people detected in the entire folder: {total_people_count}")

        # Print the absolute difference between the number of people detected and the actual count
        print(f"Mean absolute difference between the number of people detected and the actual count: {absolte_difference}")
    

# Usage
if not is_manual_testing:

    try:
        video_path = sys.argv[1]
    except IndexError:
        print("Usage: python main.py <your-video-file-path> <confidence-threshold>")
        sys.exit(1)

    try:
        input = sys.argv[2]
        confidence_threshold = float(input)
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError
    except IndexError:
        print("No confidence threshold provided, default to 0.5")
        confidence_threshold = 0.5
    except ValueError:
        print("Sorry, that's not a valid score, default to confidence threshold of 0.5")
        confidence_threshold = 0.5

    if is_time_lapse:
        track_people_in_video(video_path, frame_interval=1)
    else:
        track_people_in_video(video_path, frame_interval=30)


else:
    track_people_in_video_manual()
