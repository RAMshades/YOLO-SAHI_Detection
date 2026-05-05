# YOLO-SAHI_Detection
This work provides code that combines YOLO and SAHI for large-scale imagery.

# Instructions
## Installation
To install the required Python packages, use:

```
pip install requirements.txt
```

## Manual Installation
If you want to manually install the Ultralytics package, you can do so by going to the website and finding the conda version you want. Link provided here:

https://docs.ultralytics.com/quickstart/

Then install the other packages:

```
pip install sahi numpy pandas
```

## Base Models
Please go to Ultralytics at: https://docs.ultralytics.com/models/ and select the YOLO models you want. We specifically chose YOLO models 9-10 and have provided the specific links to the models here:

| YOLO 8 | YOLO 9 | YOLO 10 | YOLO 11 |
| -------- | -------- | -------- | -------- |
| [YOLO8L](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l.pt) | [YOLO9L](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9c.pt) | [YOLO10L](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10l.pt) | [YOLO11L](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt) |

# How to Use

### Step 1) 
Download or fork the repository and install the requirements.txt. I personally like to do so in an Anaconda environment so as not to have to deal with any package issues.

### Step 2) 
Use your favorite annotation software (e.g., MATLAB image labeler, CVAT, LabelImg, etc.) and export annotations into a YOLO format. This should provide you with (class_id, x_center, y_center, width, height). 

### Step 3) 
Edit the  [Train_Data_Slicer.py](Train_Data_Slicer.py) to the file paths and outputs you want. I point out a few of the variables below here:

> [!NOTE]
>| Variable | Description |
>| -------- | -------- | 
>|image_dir | Directory for the large image you labeled. |
> | yolo_label_dir | Directory to the txt files with the same names as the images, this should be in a yolo format. |
> | small_images_dir | New directory for the smaller labeled images. |
> | small_labels_dir | New directory for the labels of the smaller images. |
> | output_json_path | COCO JSON file of the original YOLO annotations. This is necessary for SAHI to slice the data. |
> | json_file_path | New JSON file that converts from the COCO format back to YOLO to train the YOLO models (smaller images) |
> | New_height,New_Width | Number of pixels in height and width of new, smaller images |
> | Height_overlap,Width_overlap | Amount of overlap in height and width of new, smaller images in %/100 |
> | class_names | class names and ids, e.g., 0: 'Name' |

### Step 4
Create a .yaml file of the new directory. Example below:

```
# Dataset root directory
path: Sliced_images # directory where the training/testing/and validation folders are

# Train/val/test sets: specify directories, *.txt files, or lists
train: images # images for training
val: test # images for testing

nc: 1 # number of classes
# Classes (example using 80 COCO classes)
names:
    0: Name
```

### Step 5


