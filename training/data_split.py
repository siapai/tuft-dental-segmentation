import os
import json
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Define paths
    image_dir = 'data/Radiographs/'
    mask_dir = 'data/Segmentation/teeth_mask/'

    # List files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg')])

    # Ensure the number of images and masks are equal
    assert len(image_files) == len(mask_files), "Number of images and masks should be the same"

    # Split the dataset
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files,
        mask_files,
        test_size=0.3,
        random_state=42
    )

    val_images, test_images, val_masks, test_masks = train_test_split(
        val_images,
        val_masks,
        test_size=0.5,
        random_state=42
    )

    # Create JSON structure
    train_data = [
        {
            "image": os.path.join(image_dir, img),
            "mask": os.path.join(mask_dir, msk)
        }
        for img, msk in zip(train_images, train_masks)
    ]

    val_data = [
        {
            "image": os.path.join(image_dir, img),
            "mask": os.path.join(mask_dir, msk)
        }
        for img, msk in zip(val_images, val_masks)
    ]

    test_data = [
        {
            "image": os.path.join(image_dir, img),
            "mask": os.path.join(mask_dir, msk)
        }
        for img, msk in zip(test_images, test_masks)
    ]

    combined_data = {
        'train': train_data,
        'valid': val_data,
        'test': test_data,
    }

    # Save the JSON file
    with open('data_split_2.json', 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)
