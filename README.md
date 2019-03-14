# crnn_ctc
CRNN_CTC_PyTorch
================
This version is an improvement based on [Mei](https://github.com/meijieru/crnn.pytorch) used to train CRNN model on PyTorch1.0.
Relevant paper is "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" by Shi et. al..

To run a demo, please refer to [Mei](https://github.com/meijieru/crnn.pytorch#run-demo).

Train a model
-------------
To train a model (I used SynthText90k dataset to train), check the details in the code and run like the following steps:
1. Run ``python saveAsLmdb.py``
2. Run ``python crnn_main.py --cuda``
