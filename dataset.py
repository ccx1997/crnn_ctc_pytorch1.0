import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image


class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = resizeNormalize((100, 32)) if transform is None else transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            length_key = 'length-%09d' % index
            label = eval(txn.get(label_key.encode()).decode())
            length = eval(txn.get(length_key.encode()).decode())
            label, length = torch.IntTensor(label), torch.IntTensor(length)

            text_key = 'text-%09d' % index
            text = txn.get(text_key.encode()).decode()
            
            img = self.transform(img)

        return img, label, length, text


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):

    def __init__(self):
        pass

    def __call__(self, batch):
        images, labels, lengths, text = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.cat(labels)
        lengths = torch.cat(lengths)

        return images, labels, lengths, text


if __name__ == "__main__":
    dataSet = lmdbDataset('/workspace/datasets/TR/synth90k/val')
    for i in range(5):
        img, label, length, text = dataSet[i]
