from PIL import Image
import imagehash
from pathlib import Path
import numpy as np
from scipy.io import savemat

# parameters:
cropped_images_path = 'croppedimages/path/'
output_features_name = 'features_name'

# definitions
def imhasget(file):
    """
    Returns the cropped image files from the path.
    
    Args:
        pathname (str): Directory where the cropped images are.
    """
    # Get the image hash (wavelet hashing)
    values = imagehash.whash(Image.open(file))
    
    # Convert the hash object to a hexadecimal string
    hex_hash = str(values)
    
    # Convert the hexadecimal string to a binary string
    # 'f' is '1111', '0' is '0000'. The length will be 4 * (hash length).
    binary_string = bin(int(hex_hash, 16))[2:].zfill(len(hex_hash) * 4)
    
    # Convert the binary string to a numerical binary vector (array of 0s and 1s)
    binary_vector = [int(bit) for bit in binary_string]
    
    return binary_vector
  
def getfiles(pathname):
    """
    Returns the cropped image files from the path.

    Args:
        pathname (str): Directory where the cropped images are.
    """
    path_obj = Path(cropped_images_path)
    files = list(path_obj.glob("*.png"))
    string_list = [str(path) for path in files]
    string_list.sort()
    return string_list

# run through your path and hash images.
hvalues = []
gf = getfiles(cropped_images_path)
for j in gf:
    hvalues.append(imhasget(j))

# put data into a dictionary and save as a .mat file to do unsupervised learning in MATLAB or other software.
mdic = {'hvalues': hvalues}
savemat(output_features_name+'hash.mat',mdic)
