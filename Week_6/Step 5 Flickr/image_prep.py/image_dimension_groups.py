import os
from collections import defaultdict
from PIL import Image

def group_images_by_dimensions(image_folder):
    # Dictionary to store image paths by their dimensions
    dimension_groups = defaultdict(list)
    
    # Go through each image and record its dimensions
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        try:
            with Image.open(image_path) as img:
                dimensions = img.size  # (width, height)
                dimension_groups[dimensions].append(image_name)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    
    return dimension_groups

# Replace 'path/to/your/image_folder' with the actual path to your image folder
# image_folder = '../data/flickr30k-images'
image_folder = '../data/flickr30k-images-resized'
dimension_groups = group_images_by_dimensions(image_folder)

# Print out the different groups
print("Image Groups by Dimensions:")
for dimensions, images in dimension_groups.items():
    print(f"Dimensions: {dimensions}, Count: {len(images)}")
    print("Sample images:", images[:5])  # Show a sample of 5 images per group
    print("-" * 40)
