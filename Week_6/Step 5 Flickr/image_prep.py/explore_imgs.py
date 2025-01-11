import os
from PIL import Image
import matplotlib.pyplot as plt

# Replace with the path to your Flickr image folder
image_folder = './data/flickr30k-images'

# Function to display an image given its ID
def display_image_by_id(image_id, image_folder):
    # Construct the full file path for the image
    image_path = os.path.join(image_folder, image_id)
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image {image_id} not found in {image_folder}.")
        return
    
    # Open and display the image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')  # Hide axes for a cleaner display
    plt.show()

# Test: Display an image by specifying its ID
image_id = '1029737941.jpg'  # Replace with an actual image ID from your dataset
display_image_by_id(image_id, image_folder)
