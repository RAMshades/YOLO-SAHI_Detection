from ultralytics import YOLO
import pickle

# Names of the base models
names = ['yolov8l.pt','yolov9c.pt','yolov10l.pt','yolo11l.pt']

# stepping through each base model and training
for name in names:
    model = YOLO(name)
    results = model.train(data="File.yaml", epochs=50, imgsz=1024,batch=16)

    mod_info = model.val()

    # Save the testing data results to a file
    with open(name[:-3]'_valinfo.pkl', 'wb') as f:
        pickle.dump(mod_info, f)
