# ==============================================================================
# ==============================================================================
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import contextlib2

from PIL import Image

import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
tf.flags.DEFINE_string('train_image_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '',
                       'Validation image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

#############################################################################################
def create_tf_example(frame, image_dir, category_index):
    """Converts image and annotations to a tf.Example proto.
    Args:
      frame:
        list of dicts with keys in BDD format:
        Notice that bounding box coordinates in BDD dataset are
        given as [x1, y1, x2, y2] tuples using absolute coordinates where
        x1, y1 represent the top-left corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.
      category_index: a dict containing BDD category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index_from_labelmap function.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    category_names = []
    category_ids = []
    num_annotations_skipped = 0

    image_filename = frame['name']
    image_path = os.path.join(image_dir, image_filename)
    with Image.open(image_path) as image:
        iwidth, iheight = image.size

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    #Obtain box coordinates for the image
    for label in frame['labels']:
        if 'box2d' not in label:
            num_annotations_skipped += 1
            continue
        xy = label['box2d']
        (x1, y1, x2, y2) = (xy['x1'] / iwidth, xy['y1'] / iheight, xy['x2'] / iwidth, xy['y2'] / iheight)

        #Make sure x1 == xmin y1 == ymin
        if x1 >= x2 or y1 >= y2:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x1))
        xmax.append(float(x2))
        ymin.append(float(y1))
        ymax.append(float(y2))

        #Obtain class from class name:
        for id in range(10):
            id += 1 #First id is 1 not 0
            if (category_index[id]['name'] == label['category']):
                break
            id -= 1 #Restore iterand

        category_id = int(id)
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))

    #Create feature dict with all the info
    feature_dict = {
        'image/height':
            dataset_util.int64_feature(iheight),
        'image/width':
            dataset_util.int64_feature(iwidth),
        'image/filename':
            dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, num_annotations_skipped


####################################################################################################
def _create_tf_record_from_bdd_annotations(annotations_file, image_dir, output_path, num_shards):
    """Loads BDD annotation json files and converts to tf.Record format.
    Args:
      annotations_file: JSON file containing bounding box annotations.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      num_shards: number of output file shards.
    """
    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)

        groundtruth_data = json.load(fid)
        category_index = label_map_util.create_category_index_from_labelmap('bdd_label_map.pbtxt',
                                                                            use_display_name=False)
        total_num_annotations_skipped = 0
        groundtruth_data_sub = []
        for idx, frame in enumerate(groundtruth_data):
            if idx % 10 == 0:
                groundtruth_data_sub.append(frame)

        for idx, frame in enumerate(groundtruth_data_sub):
            if idx % 100 == 0:
                tf.logging.info('On image {} of {}'.format(idx+1, len(groundtruth_data_sub)))

            tf_example, num_annotations_skipped = create_tf_example(frame, image_dir, category_index)
            total_num_annotations_skipped += num_annotations_skipped
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        tf.logging.info('Finished writing, skipped {} annotations.'.format(total_num_annotations_skipped))

############################################################################################
def main(_):
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'bdd100k_subset_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'bdd100k_subset_val.record')

  _create_tf_record_from_bdd_annotations(FLAGS.train_annotations_file, FLAGS.train_image_dir, train_output_path,
                                         num_shards=10)
  _create_tf_record_from_bdd_annotations(FLAGS.val_annotations_file, FLAGS.val_image_dir, val_output_path,
                                         num_shards=1)

if __name__ == '__main__':
    tf.app.run()