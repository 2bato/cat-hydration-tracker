import os
from PIL import Image
import imagehash

# Define input path
input_folder = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "extracted_frames"
)

# Parameters
hash_size = 8

# Initialize hashes dictionary and duplicates
hashes = {}
duplicates = []

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    try:
        with Image.open(file_path) as img:
            # Compute the perceptual hash of the image
            img_hash = imagehash.average_hash(img, hash_size=hash_size)

            if img_hash in hashes:
                duplicates.append(file_path)
            else:
                hashes[img_hash] = file_path
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Delete duplicates
if duplicates:
    for duplicate in duplicates:
        try:
            os.remove(duplicate)
        except Exception as e:
            print(f"Error removing {duplicate}: {e}")

    print(f"Duplicates deleted: {len(duplicates)}")
