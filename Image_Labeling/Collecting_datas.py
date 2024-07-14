import os

import cv2

# Parameters
labels = ['class_0', 'class_1']  # Change according to your classes
num_images_per_class = 1000  # Number of images to capture per class

# Create directories for datasets and labels
dataset_dir = 'datasets'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

for label in labels:
    label_dir = os.path.join(dataset_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture images for a specific label
def capture_images(label, num_images):
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    count = 0
    label_dir = os.path.join(dataset_dir, label)

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale (needed for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y+h, x:x+w]
            
            # Save the face ROI
            frame_filename = os.path.join(label_dir, f'{label}_{count}.jpg')
            cv2.imwrite(frame_filename, face_roi)
            count += 1
            
            # Exit if the required number of images is reached
            if count >= num_images:
                break
        
        # Display the live video
        cv2.imshow(f'Capturing {label}', frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Capture images for each label
for label in labels:
    print(f'Capturing images for {label}...')
    capture_images(label, num_images_per_class)
    print(f'Finished capturing images for {label}.')
