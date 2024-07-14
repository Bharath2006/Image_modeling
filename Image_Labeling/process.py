import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

# Parameters
image_size = (64, 64)  # Resize images to this size
labels = ['class_0', 'class_1']  # Change according to your classes
num_classes = len(labels)

# Load images and labels
x_data = []
y_data = []

for class_label, class_name in enumerate(labels):
    class_dir = os.path.join('datasets', class_name)
    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img)
        x_data.append(img_array)
        y_data.append(class_label)

# Convert to numpy arrays
x_data = np.array(x_data, dtype='float32') / 255.0  # Normalize pixel values
y_data = to_categorical(y_data, num_classes=num_classes)

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Save the preprocessed data
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump((x_train, x_test, y_train, y_test), f)
