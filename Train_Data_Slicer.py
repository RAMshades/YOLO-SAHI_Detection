import os
import json
from PIL import Image
from sahi.slicing import slice_coco

# Define paths

## large images (large initial labeled images)
image_dir = f'Sliced_images/images/train/'
yolo_label_dir = f'Sliced_images/labels/train/'

## small images (output smaller images after slicing)
small_images_dir = f'Sliced_images/images/train_smallimages/'
small_labels_dir = f'Sliced_images/labels/train_smallimageslabels/'

## Json Output paths (information for )
output_json_path = f'Sliced_images/output_coco_annotations_train.json'
json_file_path = f'Sliced_images/info_train.json'
newjson_file = json_file_path[:-4]+'_coco.json'

# Define size and overlap of smaller images
New_height,New_Width = 320,320 # Number of pixels
Height_overlap,Width_overlap = 0.3,0.3 # percentage of overlap/100

# set class names
class_names = {
    0: 'Name',
}

# Initialize COCO JSON structure
coco_format = {
    'info': {'description': 'identify small objects'},
    'licenses': [{'url': '...', 'id': 0, 'name': 'License Name'}],
    'images': [],
    'annotations': [],
    'categories': [{'id': i, 'name': name} for i, name in class_names.items()]
}

# definitions
def coco_to_yolo(json_path, output_dir):
    """
    Converts a COCO JSON annotation file to YOLO .txt format files.

    Args:
        json_path (str): Path to the COCO JSON annotation file.
        output_dir (str): Directory to save the YOLO .txt files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Map image IDs to their metadata (file_name, width, height)
    image_metadata = {img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']}
                      for img in coco_data['images']}

    # Group annotations by image ID
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Process each image and its annotations
    for image_id, image_info in image_metadata.items():
        file_name = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']

        # Construct the output file path
        base_name = os.path.splitext(file_name)[0]
        yolo_file_path = os.path.join(output_dir, f"{base_name}.txt")

        annotations = annotations_by_image.get(image_id, [])

        with open(yolo_file_path, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x_min, y_min, width_coco, height_coco]

                # Convert to normalized YOLO format
                x_min, y_min, width_coco, height_coco = bbox
                
                # Calculate YOLO center and dimensions
                x_center = (x_min + width_coco / 2) / img_width
                y_center = (y_min + height_coco / 2) / img_height
                width_yolo = width_coco / img_width
                height_yolo = height_coco / img_height
                
                # Write to file in the required format
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width_yolo:.6f} {height_yolo:.6f}\n")

    print(f"Conversion complete. YOLO files saved to {output_dir}")
    
def remove_empty_labels_and_images(labels_folder, images_folder, image_extensions=('.jpg', '.jpeg', '.png')):
    """
    Removes empty YOLO label files and their corresponding images.

    Args:
        labels_folder (str): The path to the folder containing YOLO .txt label files.
        images_folder (str): The path to the folder containing the images.
        image_extensions (tuple): A tuple of valid image file extensions.
    """
    files_to_delete = []

    # 1. Iterate through all files in the labels folder
    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_folder, label_file)

            # 2. Check if the text file is empty
            if os.path.getsize(label_path) == 0:
                print(f"Found empty label file: {label_file}")

                # 3. Identify the corresponding image
                base_name = os.path.splitext(label_file)[0]
                corresponding_image = None
                
                # Check for various image extensions
                for ext in image_extensions:
                    image_path = os.path.join(images_folder, base_name + ext)
                    if os.path.exists(image_path):
                        corresponding_image = image_path
                        break
                
                # If a corresponding image is found, add both to the delete list
                if corresponding_image:
                    files_to_delete.append((label_path, corresponding_image))
                else:
                    print(f"Warning: No corresponding image found for {label_file}. It will not be deleted.")

    if not files_to_delete:
        print("\nNo empty label files with corresponding images were found. Nothing to do. ✨")
        return

    # 4. Prompt for confirmation before deletion
    print("\nThe following files will be deleted:")
    for label, image in files_to_delete:
        print(f" - Label: {label}")
        print(f" - Image: {image}\n")

    confirmation = input("Do you want to proceed with the deletion? (yes/no): ").lower()
    if confirmation == 'yes' or confirmation == 'y':
        # 5. Remove the files
        for label_path, image_path in files_to_delete:
            os.remove(label_path)
            os.remove(image_path)
            print(f"Deleted: {label_path} and {image_path}")
        print("\nDeletion complete. All specified files have been removed.")
    else:
        print("\nDeletion canceled by user. Files were not removed.")

# Iterate through images
image_id = 0
annotation_id = 0

for img_name in os.listdir(image_dir):
    if img_name.endswith(('.png')):
        img_path = os.path.join(image_dir, img_name)
        
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        # Add image to COCO format
        coco_format['images'].append({
            'id': image_id,
            'width': width,
            'height': height,
            'file_name': img_name
        })

        # Read and convert annotations
        label_file_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(yolo_label_dir, label_file_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # Parse YOLO format
                    parts = list(map(float, line.strip().split()))
                    class_id, x_center, y_center, yolo_width, yolo_height = parts
                    
                    # Convert to COCO format
                    x_min = int((x_center - yolo_width / 2) * width)
                    y_min = int((y_center - yolo_height / 2) * height)
                    coco_width = int(yolo_width * width)
                    coco_height = int(yolo_height * height)

                    # Add annotation to COCO format
                    coco_format['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': int(class_id),
                        'bbox': [x_min, y_min, coco_width, coco_height],
                        'area': coco_width * coco_height,
                        'iscrowd': 0 # Assuming not a crowd
                    })
                    annotation_id += 1
        
        image_id += 1

# Save the COCO JSON file
with open(output_json_path, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f"Successfully converted and saved to {output_json_path}")

# slice the images up
coco_dict, coco_path = slice_coco(
    output_coco_annotation_file_name='../../../'+json_file_path[:-4],
    output_dir = small_images_dir,
    coco_annotation_file_path=output_json_path,
    image_dir=image_dir,
    slice_height=New_height,
    slice_width=New_Width,
    overlap_height_ratio=Height_overlap,
    overlap_width_ratio=Width_overlap,
)

coco_to_yolo(newjson_file, small_labels_dir)
remove_empty_labels_and_images(small_labels_dir, small_images_dir)