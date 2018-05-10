## NOTE
This implementation is fork of https://github.com/XifengGuo/CapsNet-Keras , applied to IMDB texts reviews dataset and Rotten Tomotoes dataset 



# CapsNet-Keras

A Keras implementation of CapsNet in the paper:
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)





## Requirements
- [Keras](https://github.com/fchollet/keras) 

## Usage

### Training
**Step 1.**
Install Keras:

`$ pip install keras`

**Step 2.** 
Clone this repository with ``git``.

```
$ git clone https://github.com/charlieanna/project.git
$ cd CapsNet-Keras
```

We have analyzed the capsule net on two datasets, imdb and rotten tomotoes dataset. 
You can check the results of the traning as well as the test results by using the following commands which will run the python files. 

### Analysis on the rotten tomatoes dataset.
-python rotten_cnn.py
-python rotten_capsulenet.py --model=LSTM
-python rotten_capsulenet.py --model=GRU
-python rotten_capsulenet.py --model=CuDNNLSTM
-python rotten_capsulenet.py --model=CuDNNGRU

### Analysis on the imdb dataset.
-python imdb_cnn.py
-python imdb_capsulenet.py --model=LSTM
-python imdb_capsulenet.py --model=GRU
-python imdb_capsulenet.py --model=CuDNNLSTM
-python imdb_capsulenet.py --model=CuDNNGRU

### Testing

Suppose you have trained a model using the above command, then the trained model will be
saved to `result/trained_model.h5`. Now just launch the following command to get test results.
```
$ python capsulenet.py --is_training 0 --weights result/trained_model.h5
```
It will output the testing accuracy and show the reconstructed images.
The testing data is same as the validation data. It will be easy to test on new data, 
just change the code as you want (Of course you can do it!!!)


