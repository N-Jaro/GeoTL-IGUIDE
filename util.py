# util.py

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import zipfile
import urllib.request
import shutil
import tensorflow as tf

# -------------------------------------------
# Environment Setup and File Management
# -------------------------------------------
def setup_environment_and_files():
    """
    Setup environment for CPU-only execution, install required libraries,
    download pretrained models and training history from Figshare,
    and extract files to the current working directory.
    """
    
    # Setup for CPU-only execution
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    print("Using CPU only.")
    
    # Install required libraries
    print("\nInstalling required libraries...")
    os.system("pip install timm torch torchvision 'git+https://github.com/facebookresearch/detectron2.git' --user transformers pytorch-lightning lightning-bolts segmentation-models-pytorch segmentation-models tensorflow zipfile36")
    print("All required libraries installed.")
    
    # Define the download URL and file paths
    figshare_url = "https://figshare.com/ndownloader/files/52609313"
    zip_file = "figshare_models.zip"
    extract_folder = "figshare_models"
    
    # Download the file if it doesn't already exist
    if not os.path.exists(zip_file):
        print("\nDownloading pretrained models from Figshare...")
        urllib.request.urlretrieve(figshare_url, zip_file)
        print("Download completed!")
    
    # Extract the zip file if not already extracted
    if not os.path.exists(extract_folder):
        print("\nExtracting pretrained models...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print("Extraction completed!")
    
    # Move all files to the current working directory
    print("\nMoving files to the current directory...")
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(os.getcwd(), file)
            shutil.move(src_path, dst_path)
            print(f"Moved: {file}")
    
    # Clean up the temporary extraction folder
    shutil.rmtree(extract_folder)
    print("\nCleanup completed!")
    
    # List the moved files
    print("\nFiles in current directory:")
    for file in os.listdir():
        print(file)

# -------------------------------------------
# Visualize UC Merced Land Use Dataset Images
# -------------------------------------------
def visualize_uc_merced_images(dataset_folder, num_images_per_class=2):
    """
    Visualize example images from the UC Merced Land Use Dataset.

    Args:
    - dataset_folder (str): Path to the extracted UC Merced Land Use Dataset folder.
    - num_images_per_class (int): Number of example images to show per class.
    """

    # Get the list of classes (subfolders)
    classes = sorted(os.listdir(dataset_folder))
    print(f"Found {len(classes)} classes: {classes}")

    # Set up the matplotlib figure
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, num_images_per_class, figsize=(15, num_classes * 3))
    
    # Loop through each class and display images
    for i, cls in enumerate(classes):
        class_dir = os.path.join(dataset_folder, cls)
        images = os.listdir(class_dir)
        
        # Randomly select images
        selected_images = random.sample(images, num_images_per_class)
        
        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(class_dir, img_name)
            image = Image.open(img_path)
            
            # Display the image
            ax = axes[i, j]
            ax.imshow(image)
            ax.set_title(cls)
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# -------------------------------------------
# Plot Training History
# -------------------------------------------
def plot_training_history(history):
    """
    Plots training and validation accuracy and loss curves.
    
    Parameters:
        history: Training history object returned by model.fit()
    """
    # Plot accuracy
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.show()

# -------------------------------------------
# Plot Model Predictions
# -------------------------------------------
def plot_predictions(model, generator, class_labels, num_images=9):
    """
    Visualizes model predictions along with true labels.
    
    Parameters:
        model: Trained model object
        generator: ImageDataGenerator object for validation/test data
        class_labels: List of class labels
        num_images: Number of images to display (default: 9)
    """
    # Get a batch of images and labels
    images, labels = next(generator)
    predictions = model.predict(images)
    
    # Plot predictions
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        pred_label = class_labels[np.argmax(predictions[i])]
        true_label = class_labels[np.argmax(labels[i])]
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
