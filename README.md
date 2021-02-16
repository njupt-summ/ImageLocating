# ImageLocating
- The dataset E-DailyMail can be constructed by https://github.com/jingqiangchen/multi-modal. And the folder datas is a example sampled from a news.
- Here you can get files [ilsvrc_2012_mean.npy](https://github.com/BVLC/caffe/blob/master/python/caffe/imagenet/ilsvrc_2012_mean.npy), [VGG_ILSVRC_19_layers.prototxt](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md) and [VGG_ILSVRC_19_layers.caffemodel](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md) in VGG19.py.

# Requirements

- Python 3
- Tensorflow 1.14
- gensim 3.8.1
- openCV
- Tqdm
- [caffe](https://github.com/BVLC/caffe)

# Training the model

python3 training.py --num_epochs 10 --batch_size 50 --learning_rate 0.0001
