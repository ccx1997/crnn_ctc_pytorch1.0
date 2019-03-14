import os
import lmdb
import cv2
import numpy as np
import imghdr
import utils


def checkImageIsValid(imageBin):
    """imageBin: image content represented by binary string like  b'\x89PNG\r\n\x1a\n\x00\x00'"""
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print('image open, Exception', e)
        return False
    try:
        imgH, imgW = img.shape[0], img.shape[1]
    except Exception:
        return False
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in iter(cache.items()):
            if not isinstance(v, bytes):
                v = v.encode()
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, root_dir, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    converter = utils.strLabelConverter(alphabet, ignore_case=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = os.path.join(root_dir, imagePathList[i])
        text = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        if imghdr.what(imagePath) != 'jpeg':
            print('{} is not jpeg'.format(imagePath))
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        label, length = converter.encode(text)
        label, length = str(label.tolist()), str(length.tolist())

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        lengthKey = 'length-%09d' % cnt
        textKey = 'text-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        cache[lengthKey] = length
        cache[textKey] = text
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 5000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    pass
