import os,cv2
import scipy.io as sio
import numpy as np
import torch
import torchvision.transforms as transforms
from models import crnn
import utils
import time


def testset_gt(root='/workspace/datasets/TR/IIIT5K/'):
    """To get a txt file consisting of file name and string without lexicon."""
    mat_file = 'testdata.mat'
    txt_file = 'gt.txt'
    mat = sio.loadmat(os.path.join(root, mat_file))
    testdata = mat['testdata']
    _, n = testdata.shape
    txt_file = os.path.join(root, txt_file)
    f = open(txt_file, 'w')
    for i in range(n):
        entry = testdata[0, i]
        img_name = entry[0][0]
        label = entry[1][0]
        line = img_name + ',' + label + '\n'
        f.writelines(line)
    f.close()


def mynet(param_file='./data/crnn.pth'):
    net = crnn.CRNN(imgH=32, nc=1, nclass=37, nh=256)
    net.load_state_dict(torch.load(param_file))
    return net


def img_transf(image):
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w / h * 32), 32))
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    tsfm = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = tsfm(image)
    return image.unsqueeze(0)


def main(root, param_file, error_file='error_samples.txt', alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    device = torch.device("cuda:0")
    net = mynet(param_file)
    net.eval()
    net.to(device)
    converter = utils.strLabelConverter(alphabet)
    txt_file = os.path.join(root, 'gt.txt')
    f = open(txt_file, 'r')
    error_f = open(error_file, 'w')
    error_f.writelines(param_file + '\n')
    cnt = 0
    n_pos = 0
    time1 = time.clock()
    for line in f:
        line = line.strip().split(',')
        img_name = os.path.join(root, line[0])
        label = line[1].lower()
        image = cv2.imread(img_name, 0)
        image = img_transf(image)
        image = image.to(device)
        with torch.no_grad():
            pred = net(image)
        _, pred = pred.max(2)
        pred = pred.transpose(1, 0).contiguous().view(-1)
        pred_size = torch.IntTensor([pred.size(0)])
        sim_pred = converter.decode(pred.data, pred_size.data, raw=False)
        cnt += 1
        if sim_pred == label:
            n_pos += 1
        else:
            raw_pred = converter.decode(pred.data, pred_size.data, raw=True)
            error_f.writelines(img_name + ': ' + '%-20s => %-20s' % (raw_pred, sim_pred) + '\n')
    time2 = time.clock()
    s1 = 'Accuracy in test set is %.1f %%' % (n_pos / cnt * 100)
    s2 = "Processing 1 image costs around %.1f ms." % ((time2 - time1) * 1000 / cnt)
    error_f.writelines(s1 + '\n' + s2)
    error_f.close()
    f.close()
    print(s1)
    print(s2)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    root = '/workspace/datasets/TR/IIIT5K/'
    param_file = './param/crnn1.pth'
    # testset_gt(root)
    main(root, param_file)
