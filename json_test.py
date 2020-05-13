import os
import json
import time
from PIL import Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

label_path = 'F:/BDBerkeley/bdd100k/labels/bdd100k_labels_images_val.json'
image_dir  = 'F:/BDBerkeley/bdd100k/images/100k/val'
category_index = label_map_util.create_category_index_from_labelmap('bdd_label_map.pbtxt', use_display_name=False)
gt = json.load(open(label_path, 'r'))

xmin = []
xmax = []
ymin = []
ymax = []
category_names = []
category_ids = []

for frame in gt:
    image_filename = frame['name']
    image_path = os.path.join(image_dir,image_filename)
    image = Image.open(image_path)
    iwidth, iheight = image.size

    for label in frame['labels']:
        if 'box2d' not in label:
            continue
        xy = label['box2d']
        (x1, y1, x2, y2) = (xy['x1']/iwidth, xy['y1']/iheight, xy['x2']/iwidth, xy['y2']/iheight)
        if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
            continue

        for id in range(10):
            id += 1
            if (category_index[id]['name'] == label['category']):
                break
            id -= 1
        print(id)

        xmin.append(float(x1))
        xmax.append(float(x2))
        ymin.append(float(y1))
        ymax.append(float(y2))

        category_id = int(id)
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))

time.sleep(100)