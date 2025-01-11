import os
from PIL import Image
import pandas as pd

# Define the path to your image folder
image_folder = './data/flickr30k-images'

# Initialize a list to store information about each image
image_data = []

# Loop through each image in the folder
for image_name in os.listdir(image_folder):
    # Check if the file is an image
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Open the image
        image_path = os.path.join(image_folder, image_name)
        with Image.open(image_path) as img:
            width, height = img.size
            # Determine orientation
            orientation = "landscape" if width > height else "portrait"
            
            # Append data to list
            image_data.append({
                'Image Name': image_name,
                'Width': width,
                'Height': height,
                'Orientation': orientation
            })

# Convert list to DataFrame
image_df = pd.DataFrame(image_data)

# Generate statistics table
orientation_stats = image_df.groupby('Orientation').size().reset_index(name='Count')
size_stats = image_df[['Width', 'Height']].describe()

# Display the data and statistics tables
print("Orientation Counts:")
print(orientation_stats)
print("\nImage Size Statistics:")
print(size_stats)

# Save the detailed table to a CSV file if needed
image_df.to_csv("image_orientation_and_size_stats.csv", index=False)
