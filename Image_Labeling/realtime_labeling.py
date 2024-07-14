import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Parameters
image_size = (64, 64)  # Same size as used during training
labels = [ 'class_1']  # Change according to your classes
specific_label = 'class_0'  # Specify the specific class label to match

# Load the trained model
model = load_model('image_classification_model.h5')

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess a frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, image_size)
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype('float32') / 255.0
    return frame

# Function to save the face image
def save_face_image(face_roi, save_path):
    cv2.imwrite(save_path, face_roi)

# Create directory for captured faces if it doesn't exist
save_dir = 'captured_faces'
os.makedirs(save_dir, exist_ok=True)

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    
    # Convert the frame to grayscale (needed for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Initialize variables to track the detected face with the largest area
    largest_area = 0
    largest_face = None
    
    for (x, y, w, h) in faces:
        # Calculate the area of the current face
        area = w * h
        
        # Keep track of the largest face found
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)
    
    # If a face is detected, process it
    if largest_face is not None:
        x, y, w, h = largest_face
        
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face ROI
        preprocessed_frame = preprocess_frame(face_roi)
        
        # Ensure the frame was preprocessed correctly
        if preprocessed_frame.shape != (1, 64, 64, 3):
            print(f"Unexpected shape for preprocessed frame: {preprocessed_frame.shape}")
            continue
        
        # Predict the class of the face ROI
        predictions = model.predict(preprocessed_frame)
        
        # Debugging info
        print(f"Predictions: {predictions}")
        
        class_idx = np.argmax(predictions, axis=1)[0]
        
        # Debugging info
        print(f"Predicted class index: {class_idx}")
        
        if class_idx >= len(labels):
            print(f"Invalid class index {class_idx}. Predictions: {predictions}")
            continue
        
        predicted_label = labels[class_idx]
        
        # Determine the color of the rectangle based on the predicted class
        if predicted_label == specific_label:
            box_color = (0, 255, 0)  # Green for specific class
        else:
            box_color = (0, 0, 255)  # Red for other classes
            
            # Save the face image with red box
            save_path = f'{save_dir}/{predicted_label}_{len(os.listdir(save_dir)) + 1}.jpg'
            save_face_image(face_roi, save_path)
        
        # Display the predicted class on the frame
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
        
        # Draw a rectangle around the face with the determined color
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
    
    # Display the live video with detections
    cv2.imshow('Live Video', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
