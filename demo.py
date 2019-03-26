import torch
import utils
import dataset
from PIL import Image
import time
import os
import argparse

import models.crnn as crnn


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='./data/demo.png', help='The name of image to be tested.')
opt = parser.parse_args()

model_path = './data/crnn1.pth'
# model_path = './param/CRNNCTC.pth'
img_path = opt.img
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

start = time.time()
converter = utils.strLabelConverter(alphabet)

image = Image.open(img_path).convert('L')
w, h = image.size
transformer = dataset.resizeNormalize((int(w/h*32), 32))
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())

model.eval()
with torch.no_grad():
    preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
print("Took %.2f s" % (time.time() - start))
