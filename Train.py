from ultralytics import YOLO
import pickle

# Parameters
model_name = 'yolov8l.pt'
yaml_file = 'File.yaml'
perf_file = 'valinfo.pkl'

## model training parameters
epochs = 50
imgsz = 1024
batch = 16

## Test and save performance curves
test_model = True
output_test_file = 'valinfo.pkl'

# Load in the base model
model = YOLO('yolo11l.pt')

# Train the model and return the results
results = model.train(data="File.yaml", epochs=epochs, imgsz=imgsz,batch=batch)

if test_model: 
    # test the model
    mod_info = model.val()

    # Save the testing data results to a file
    with open(output_test_file, 'wb') as f:
        pickle.dump(mod_info, f)
