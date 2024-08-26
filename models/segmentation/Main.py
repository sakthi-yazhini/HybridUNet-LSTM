import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import cv2

# Define the CVA-Net Model for Classification
class CVA_Net(nn.Module):
    def __init__(self):
        super(CVA_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, 2)  # Output 2 classes: benign and malignant

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the U-Net Model for Segmentation
def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up4 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3)
    merge4 = layers.Concatenate()([conv2, up4])
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv4)
    merge5 = layers.Concatenate()([conv1, up5])
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Prepare Data for Classification
def prepare_classification_data(data_dir, img_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

# Train and Evaluate the CVA-Net Model
def train_and_evaluate_classification_model(train_loader, val_loader, num_epochs=10):
    device = torch.device('cpu')  # Use 'cuda' if you have a GPU
    model = CVA_Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    no_improve = 0
    patience = 3

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = np.mean(all_labels == all_preds)
    classification_rep = classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'],
                                               output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Prepare Data for Segmentation
def prepare_segmentation_data(data_dir, img_size=(256, 256)):
    images = []
    masks = []

    for folder in ['benign', 'malignant']:
        folder_path = os.path.join(data_dir, folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                img_path = os.path.join(folder_path, file_name)
                mask_path = img_path  # Modify as necessary for your mask file structure

                img = load_img(img_path, target_size=img_size, color_mode='rgb')
                img_array = img_to_array(img) / 255.0
                images.append(img_array)

                mask = load_img(mask_path, target_size=img_size, color_mode='grayscale')
                mask_array = img_to_array(mask) / 255.0
                masks.append(mask_array)

    images = np.array(images)
    masks = np.array(masks)

    split_index = int(0.8 * len(images))
    train_images, val_images = images[:split_index], images[split_index:]
    train_masks, val_masks = masks[:split_index], masks[split_index:]

    return (train_images, train_masks), (val_images, val_masks)

# Train the U-Net Model
def train_unet_model(train_images, train_masks, val_images, val_masks, epochs=10):
    unet = unet_model()
    history = unet.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=epochs, batch_size=8)
    unet.save('unet_model.keras')  # Save the model in Keras format
    return unet, history

# Visualize Segmentation Results using U-Net with Bounding Boxes
def visualize_segmentation(image_path, unet_model, cva_net_model, threshold=0.5):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict the mask
    pred_mask = unet_model.predict(img_array)
    pred_mask = (pred_mask[0, :, :, 0] > threshold).astype(np.uint8)

    # Convert image and mask to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    pred_mask_cv = pred_mask * 255
    pred_mask_cv = np.uint8(pred_mask_cv)

    # Find contours in the predicted mask
    contours, _ = cv2.findContours(pred_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on the original image if malignant
    img_with_boxes = img_cv.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display classification result
    cva_net_model.eval()
    img_for_classification = img_to_array(load_img(image_path, target_size=(256, 256))) / 255.0
    img_for_classification = np.expand_dims(img_for_classification, axis=0)
    img_for_classification = torch.tensor(img_for_classification).permute(0, 3, 1, 2).float()

    with torch.no_grad():
        output = cva_net_model(img_for_classification)
        _, predicted_class = torch.max(output, 1)
        class_label = 'Malignant' if predicted_class.item() == 1 else 'Benign'

    # Plot results
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(img)

    plt.subplot(1, 4, 2)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title('Overlay with Bounding Boxes')
    plt.imshow(img_with_boxes)

    plt.subplot(1, 4, 4)
    plt.title(f'Classification: {class_label}')
    plt.axis('off')

    plt.show()

# Main function to run the classification and segmentation tasks
if __name__ == "__main__":
    # Paths and Parameters
    data_dir = '/Users/sakthiyazhini/PycharmProjects/segmentation/dataset'
    img_size = (256, 256)

    # Prepare and train the classification model
    train_loader, val_loader = prepare_classification_data(data_dir, img_size)
    train_and_evaluate_classification_model(train_loader, val_loader, num_epochs=10)

    # Prepare and train the segmentation model
    (train_images, train_masks), (val_images, val_masks) = prepare_segmentation_data(data_dir, img_size)
    unet_model, _ = train_unet_model(train_images, train_masks, val_images, val_masks, epochs=10)

    # Initialize and train CVA-Net model
    cva_net_model = CVA_Net()
    train_loader, val_loader = prepare_classification_data(data_dir, img_size)
    train_and_evaluate_classification_model(train_loader, val_loader, num_epochs=10)

    # Save the CVA-Net model
    torch.save(cva_net_model.state_dict(), 'segmentation_model.pth')

    # Load CVA-Net model for classification
    cva_net_model = CVA_Net()
    cva_net_model.load_state_dict(torch.load('segmentation_model.pth', map_location=torch.device('cpu')))
    cva_net_model.eval()

    # Image path for simple testing
    image_path = '/Users/sakthiyazhini/PycharmProjects/segmentation/dataset/benign/000000.png'  # Replace with your image path
    visualize_segmentation(image_path, unet_model, cva_net_model)
