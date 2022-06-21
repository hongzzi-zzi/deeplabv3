#%%
import json
from glob import glob

import numpy as np
from PIL import Image
from pycocotools import mask
from skimage import measure
import numpy as np
#%%
bg=np.array(Image.open('/home/h/Desktop/data/random/train/b_label/b_label1_001.png').resize((512, 512)).split()[-1], dtype=np.uint8)
teeth=np.array(Image.open('/home/h/Desktop/data/random/train/t_label/t_label1_001.png').resize((512, 512)).split()[-1], dtype=np.uint8)
#%%
ground_truth_binary_mask=np.array(bg)
fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
ground_truth_area = mask.area(encoded_ground_truth)
ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
contours = measure.find_contours(ground_truth_binary_mask, 0.5)

annotation = {
        "segmentation": [],
        "area": ground_truth_area.tolist(),
        "iscrowd": 0,
        "image_id": 123,
        "bbox": ground_truth_bounding_box.tolist(),
        "category_id": "bg",
        "id": 1
    }

for contour in contours:
    contour = np.flip(contour, axis=1)
    segmentation = contour.ravel().tolist()
    annotation["segmentation"].append(segmentation)
    
print(json.dumps(annotation, indent=4))

#%%
ground_truth_binary_mask=np.array(teeth)
fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
ground_truth_area = mask.area(encoded_ground_truth)
ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
contours = measure.find_contours(ground_truth_binary_mask, 0.5)

annotation = {
        "segmentation": [],
        "area": ground_truth_area.tolist(),
        "iscrowd": 0,
        "image_id": 123,
        "bbox": ground_truth_bounding_box.tolist(),
        "category_id": "teeth",
        "id": 1
    }

for contour in contours:
    contour = np.flip(contour, axis=1)
    segmentation = contour.ravel().tolist()
    annotation["segmentation"].append(segmentation)
    
print(json.dumps(annotation, indent=4))
# %%
import numpy as np
from imantics import Polygons, Mask

# This can be any array
array = np.ones((100, 100))

polygons = Mask(bg).polygons()

print(polygons.points)
print(polygons.segmentation)
# %%
