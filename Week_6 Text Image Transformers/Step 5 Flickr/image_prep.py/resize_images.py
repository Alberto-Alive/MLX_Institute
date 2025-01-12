import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size=(224, 224)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    count = 0
    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        # Build the full path to the input image
        input_path = os.path.join(input_folder, filename)
        
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            with Image.open(input_path) as img:
                # Resize the image using Image.Resampling.LANCZOS
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Build the full path to save the output image
                output_path = os.path.join(output_folder, filename)
                
                # Save the resized image
                resized_img.save(output_path)
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} images.")

# Example usage
input_folder = '../data/flickr30k-images'
output_folder = '../data/224by224'
resize_images(input_folder, output_folder)


            
