from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
import pickle
from sahi import AutoDetectionModel
from pathlib import Path
import numpy as np
from PIL import Image

# parameters to set

## Large Images Dir
dir_image_path = "Sliced_images/test/"

## Trained YOLO model parameters
model_weights = 'runs/detect/train/weights/best.pt'
det_threshold = 0.5
device = "cpu"

## sliced prediction parameters
Image_height,Image_width = 320,320
Overlap_height,Overlap_width = 0.2,0.2
prediction_dir = '/'

## crop objects?
crop_image = False
cropped_image_dir = '/'


# Find all files recursively
directory_path = Path(dir_image_path)  # Replace with your directory path
all_files = list(directory_path.rglob("*")) 

# get info about the center location
def getcenter(info):
    """
    Returns the information about the bounding box from YOLO prediction.

    Args:
        info (list): Sliced inference prediction from SAHI
    """
    id = []
    centerx = []
    centery = []
    score = []
    area = []
    width =[]
    height =[]
    if any(info):
        for i in info:
            id.append(i['category_id'])
            centerx.append(i['bbox'][0])
            centery.append(i['bbox'][1])
            score.append(i['score'])
            area.append(i['area'])
            width.append(i['bbox'][2])
            height.append(i['bbox'][3])
    return id,centerx,centery,area,width,height,score

def cropimage(im,x,y,w,h,name,folder):
    """
    Crops and saves the small object from the bounding box from YOLO prediction.

    Args:
        im (str): PIL image
        x (tuple): center x position of bounding box
        y (tuple): center y position of bounding box
        w (tuple): width of bounding box
        h (tuple): height of bounding box
        name (str): output name of cropped image
        folder (str): directory to save the cropped image
        info (list): Sliced inference prediction from SAHI
    """
    startpointx=int(np.round(x-w/2))
    startpointy=int(np.round(y-h/2))
    endpointx =int(np.round(x+w/2))
    endpointy =int(np.round(y+h/2))

    cropped_img = im.crop((startpointx, startpointy, endpointx, endpointy))
    cropped_img.save(folder+name)

# load in detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_weights,
    confidence_threshold=det_threshold,
    device=device,  # or 'cuda:0'
)

# run through the files in the folder to predict
# note: these should be the large images not the small sliced images, the below code splits the large image and does the prediction
for file in all_files:
    n = str(file)
    results = get_sliced_prediction(
        n,
        detection_model,
        slice_height=Image_height,
        slice_width=Image_width,
        overlap_height_ratio=Overlap_height,
        overlap_width_ratio=Overlap_width,
    )

    ind = n.rfind('\\') # find the last / in the file name
    # save prediction output to pickle
    with open(prediction_dir+n[ind+1:-4]+'_sahi_pred/'+str(file), 'wb') as f:
            pickle.dump(results.object_prediction_list, f)

    # if we set the crop image above
    if crop_image:
        id,centerx,centery,area,width,height,score = getcenter(results.to_coco_annotations())
        im = Image.open(n)
        for i in range(0,len(id)):
                name = cropped_image_dir+n[ind+1:-4]+f"_obj_{i:03}.png"
                cropimage(im,centerx[i],centery[i],width[i],height[i],name)

