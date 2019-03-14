import imghdr
import cv2 
import os
from PIL import Image
import numpy as np

'''Remove the corrupted images in the dataset.
'''


image_dir = '/workspace/datasets/TR/synth90k/90kDICT32px'
image_tags_file = os.path.join(image_dir, 'annotation_val.txt')    # File to be filtered
filter_file = os.path.join(image_dir, 'annotation_val1.txt')    # File to save the correct samples

fo_filted = open(filter_file, 'w')
with open(image_tags_file) as fo:
  for line in fo:
    image_name = line.split()[0].replace('./', '')
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
      print('{} not exist'.format(image_path))
      continue 
    
    if imghdr.what(image_path) != 'jpeg':
      print('{} is not jpeg'.format(image_path))
      continue
    
    try:
      image = Image.open(image_path)
      image = np.array(image, dtype=np.float32)
      image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    except Exception as e:
      print('image open, Exception', e, image_path)
      continue
    
    if len(image.shape) != 3:
      print('image shape is not 3 {}'.format(image_path))
      continue

    if image is None:
      print('read {} is None'.format(image_path))
      continue
      

    fo_filted.write('{}\n'.format(image_name))
fo_filted.close()

