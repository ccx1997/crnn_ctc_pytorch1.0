import argparse
import torch
import torch.optim as optim
import torch.utils.data
import os
import utils
import dataset

import models.crnn as crnn

import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', default='/workspace/datasets/TR/synth90k/train', help='path to dataset')
parser.add_argument('--valroot', default='/workspace/datasets/TR/synth90k/val', help='path to dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1.0, help='learning rate, default=1.0')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=250, help='Interval to be displayed')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'data'
os.system('mkdir {0}'.format(opt.experiment))

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

collate = dataset.alignCollate()
train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    collate_fn=collate,
)
test_dataset = dataset.lmdbDataset(root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))
test_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers), collate_fn=collate)

nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = torch.nn.CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn), strict=False)
print(crnn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
crnn = crnn.to(device)
criterion = criterion.to(device)

# loss averager
loss_avg = utils.averager()

# setup optimizer
# optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, weight_decay=1e-4)
optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.3)


def val(net, data_loader, criterion, max_iter=100):
    print('Start val')

    net.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for _ in range(max_iter):
        data = val_iter.next()
        cpu_images, text, length, cpu_texts = data
        image = cpu_images.to(device)
        batch_size = cpu_images.size(0)

        with torch.no_grad():
            preds = crnn(image)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length)
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %.2f%%' % (loss_avg.val(), accuracy*100))


def trainBatch():
    crnn.train()
    data = train_iter.next()
    image, text, length, _ = data
    image = image.to(device)
    image.requires_grad_()
    batch_size = image.size(0)

    preds = crnn(image)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)
    cost = criterion(preds, text, preds_size, length)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(opt.epoch):
    train_iter = iter(train_loader)
    i = 0

    while i < len(train_loader):
        scheduler.step()
        if optimizer.param_groups[0]['lr'] < 1e-4:
            optimizer.param_groups[0]['lr'] = 1e-2
        time0 = time.time()
        cost = trainBatch()
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] lr: %.4f Loss: %f Time: %f s' %
                  (epoch, opt.epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], loss_avg.val(), time.time()-time0))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_loader, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(crnn.state_dict(), '{0}/crnn1.pth'.format(opt.experiment))
