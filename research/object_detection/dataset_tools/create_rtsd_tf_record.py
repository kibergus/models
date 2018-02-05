# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Creates TFRecords of Open Images dataset for object detection.

Example usage:
  python object_detection/dataset_tools/create_oid_tf_record.py \
    --input_annotations_csv=/path/to/input/annotations-human-bbox.csv \
    --input_images_directory=/path/to/input/image_pixels_directory \
    --input_label_map=/path/to/input/labels_bbox_545.labelmap \
    --output_tf_record_path_prefix=/path/to/output/prefix.tfrecord

CSVs with bounding box annotations and image metadata (including the image URLs)
can be downloaded from the Open Images GitHub repository:
https://github.com/openimages/dataset

This script will include every image found in the input_images_directory in the
output TFRecord, even if the image has no corresponding bounding box annotations
in the input_annotations_csv.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io

import contextlib2
import pandas as pd
import numpy as np
import tensorflow as tf
import PIL.Image as pil

from sklearn.utils import shuffle
from object_detection.dataset_tools import oid_tfrecord_creation
from object_detection.utils import label_map_util

tf.flags.DEFINE_string('input_annotations_csv', None,
                       'Path to CSV containing image bounding box annotations')
tf.flags.DEFINE_string('input_images_directory', None,
                       'Directory containing the image pixels '
                       'downloaded from the OpenImages GitHub repository.')
tf.flags.DEFINE_string('input_label_map', None, 'Path to the label map proto')
tf.flags.DEFINE_string(
    'output_tf_record_path_prefix', None,
    'Path to the output TFRecord. The shard index and the number of shards '
    'will be appended for each output shard.')
tf.flags.DEFINE_integer('num_shards', 100, 'Number of TFRecord shards')
tf.flags.DEFINE_integer('validation_set_size', '20000', 'Number of images to'
    'be used as a validation set.')

FLAGS = tf.flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = [
      'input_annotations_csv', 'input_images_directory', 'input_label_map',
      'output_tf_record_path_prefix'
  ]
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  label_map = label_map_util.get_label_map_dict(FLAGS.input_label_map)
  all_annotations = pd.read_csv(FLAGS.input_annotations_csv)
  all_images = tf.gfile.Glob(
      os.path.join(FLAGS.input_images_directory, '*.jpg'))
  all_image_ids = [os.path.basename(v) for v in all_images]
  all_image_ids = pd.DataFrame({'filename': all_image_ids})
  all_annotations = pd.concat([all_annotations, all_image_ids])

  tf.logging.log(tf.logging.INFO, 'Found %d images...', len(all_image_ids))

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = oid_tfrecord_creation.open_sharded_output_tfrecords(
        tf_record_close_stack, FLAGS.output_tf_record_path_prefix + "_train",
        FLAGS.num_shards)
    test_tfrecords = oid_tfrecord_creation.open_sharded_output_tfrecords(
        tf_record_close_stack, FLAGS.output_tf_record_path_prefix + "_test",
        FLAGS.num_shards)

    for counter, image_data in enumerate(shuffle(all_annotations.groupby('filename'))):
      tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 1000,
                             counter)

      image_id, image_annotations = image_data
      image_annotations = image_annotations.copy()
      image_path = os.path.join(FLAGS.input_images_directory, image_id)
      with tf.gfile.Open(image_path) as image_file:
        encoded_image = image_file.read()

      image = pil.open(io.BytesIO(encoded_image))
      image = np.asarray(image)
      width = float(image.shape[1])
      height = float(image.shape[0])

      image_annotations['XMin'] = image_annotations['x_from'] / width
      image_annotations['XMax'] = (image_annotations['x_from'] + image_annotations['width']) / width
      image_annotations['YMin'] = image_annotations['y_from'] / height
      image_annotations['YMax'] = (image_annotations['y_from'] + image_annotations['height']) / height
      image_annotations['ImageID'] = image_annotations['filename']
      image_annotations['LabelName'] = image_annotations['sign_class']

      tf_example = oid_tfrecord_creation.tf_example_from_annotations_data_frame(
          image_annotations, label_map, encoded_image)
      if tf_example:
        shard_idx = counter % FLAGS.num_shards
        if counter < FLAGS.validation_set_size:
            test_tfrecords[shard_idx].write(tf_example.SerializeToString())
        else:
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())


if __name__ == '__main__':
  tf.app.run()
