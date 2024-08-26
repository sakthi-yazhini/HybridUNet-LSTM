import numpy as np
import os
import cv2
from PIL import Image

# Define the data directory and image size
image_directory = 'Dataset/'
benign_folder = os.path.join(image_directory, 'benign')
malignant_folder = os.path.join(image_directory, 'malignant')
INPUT_SIZE = 64

def get_image_files_from_subfolders(directory):
    image_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                image_files.append(os.path.join(subdir, file))
    return image_files

no_tumor_images = get_image_files_from_subfolders(benign_folder)
yes_tumor_images = get_image_files_from_subfolders(malignant_folder)

print('benign: ', len(no_tumor_images))
print('malignant: ', len(yes_tumor_images))

dataset = []
label = []

# Create labels
for image_path in no_tumor_images:
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    dataset.append(np.array(image))
    label.append(0)

for image_path in yes_tumor_images:
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    dataset.append(np.array(image))
    label.append(1)

dataset = np.array(dataset)
label = np.array(label)

# Save dataset and labels
np.save('dataset.npy', dataset)
np.save('labels.npy', label)
print('Dataset and labels saved.')
