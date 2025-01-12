# import pandas as pd


# df = pd.read_csv('./data/flickr_annotations_30k.csv')

# for index, row in df.iterrows():
#     captions = eval(row['raw'])
#     image_id = row['filename']
#     print(captions, image_id)
#     for caption in captions:
#         print(f"Image id: {image_id} and caption {caption}")
#     break


from PIL import Image
import os

class LoadPatchCaptionTrainPairs:
    def __init__(self, image_folder_path, captions_dict, output_dir, patch_size=16):
        self.image_folder_path = image_folder_path
        self.captions_dict = captions_dict  # Dictionary with {image_filename: [caption1, caption2, ...]}
        self.output_dir = output_dir  # Directory to save patches
        self.patch_size = patch_size

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_and_save_patches(self, image, image_id):
        image_width, image_height = image.size
        patches = []
        
        # Calculate number of patches
        num_patches_x = image_width // self.patch_size
        num_patches_y = image_height // self.patch_size
        patch_counter = 0

        # Loop through the image and extract patches
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                # Calculate the position of each patch
                left = x * self.patch_size
                top = y * self.patch_size
                right = left + self.patch_size
                bottom = top + self.patch_size

                # Crop the patch
                patch = image.crop((left, top, right, bottom))

                # Define a unique filename for each patch
                patch_filename = f"{image_id}_patch_{patch_counter}.jpg"
                patch_path = os.path.join(self.output_dir, patch_filename)

                # Save the patch as an image file
                patch.save(patch_path)
                
                patches.append(patch_path)  # Store the path for pairing with captions
                patch_counter += 1

        return patches

    def get_training_pairs(self):
        training_pairs = []

        # Loop through each image file in the folder
        for image_filename in os.listdir(self.image_folder_path):
            image_path = os.path.join(self.image_folder_path, image_filename)

            # Check if the file is an image
            if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(image_path)
                
                # Get a unique image ID from the filename (e.g., image_id_11 from 'image_id_11.jpg')
                image_id = os.path.splitext(image_filename)[0]

                # Generate and save patches
                patch_paths = self.generate_and_save_patches(image, image_id)

                # Get captions for the current image
                captions = self.captions_dict.get(image_filename, [])

                # Create a pair of each patch path with each caption
                for caption in captions:
                    for patch_path in patch_paths:
                        training_pairs.append((patch_path, caption))
        
        return training_pairs

# Example usage
image_folder_path = './data/224by224'   # Folder containing images
output_dir = './data/8by8_patches'      # Directory to save patches
captions_dict = {
    'image_id_11.jpg': ["caption_1_of_image_id_11", "caption_2_of_image_id_11"],
    'image_id_12.jpg': ["caption_1_of_image_id_12", "caption_2_of_image_id_12"],
    # Add more image filenames and their captions here
}

# Create an instance of the class
loader = LoadPatchCaptionTrainPairs(image_folder_path, captions_dict, output_dir, patch_size=16)

# Get training pairs of patches and captions
training_pairs = loader.get_training_pairs()

# Example: Display the number of pairs and show one pair
print(f"Total training pairs: {len(training_pairs)}")
print("Example pair:", training_pairs[0])
