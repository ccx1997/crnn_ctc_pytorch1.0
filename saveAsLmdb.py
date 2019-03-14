import os
from create_dataset import createDataset


def synth_90k():
    root_dir = '/workspace/datasets/TR/synth90k/90kDICT32px/'
    train_txt_name = 'annotation_train1.txt'
    val_txt_name = 'annotation_val1.txt'

    def getList(txt_name):
        txt_name = os.path.join(root_dir, txt_name)
        if not os.path.exists(txt_name):
            raise FileNotFoundError('%s does not exist' % txt_name)
        img_path_list = []
        text_list = []
        with open(txt_name, 'r') as f:
            for line in f:
                line = line.split()[0].replace('./', '')
                left = line.index('_') + 1
                right = line.index('_', left)
                text = line[left:right]
                img_path_list.append(line)
                text_list.append(text)
        return img_path_list, text_list

    trainImgList, trainTextList = getList(train_txt_name)
    createDataset('/workspace/datasets/TR/synth90k/train', trainImgList, trainTextList, root_dir, checkValid=True)
    valImgList, valTextList = getList(val_txt_name)
    createDataset('/workspace/datasets/TR/synth90k/val', valImgList, valTextList, root_dir, checkValid=True)


if __name__ == "__main__":
    synth_90k()
