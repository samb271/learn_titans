# Unofficial implementation of Titans
Compact and unofficial implementation of *Titans: Learning to Memorize at Test-Time*. Missing a few things, but prioritized readability for those wishing to have code to support the reading of the paper. 
## 1. Getting started
First, install the packages in a virtual environment. Code has uniquely been tested with Python 3.12.
#### 1.1 Dependencies
```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
#### 1.2 Dataset
Out of the box, the code only supports the WikiText-103 dataset. It's relatively small, requiring less than 600MB of storage space. Create a *data* folder at root and enter it:
```
mkdir data
cd data
```

Then, download the dataset using this link:
```
wget https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz
```

Unzip:
```
tar -xzvf wikitext-103.tar.gz
```

And you should be good to go. Confirm that the following three datasets are within the folder *./data/wikitext-103/*
```
|- wiki.test.tokens
|- wiki.train.tokens
|- wiki.valid.tokens
```

That's it!

## 2. Training
#### 2.1 Titans
To train Titans, you can execute the following command:
```
python scripts/titans/train.py --config_path=scripts/titans/conf.yaml 
```

This will start training Titans with test-time memory on the dataset. You can play around with the configuration file in the *scripts/titans/* folder to try different hyperparameters. 

#### 2.2 Memory

If you want to get a feel for how the memory module works, you can execute the memory file as a standalone:
```
python models/memory.py
```

You will see how the outer loop (the language model) and inner loop (the test-time memory) losses evolve independently from eachother. Have fun dissecting this code - once you understand what's contained within that single python file, you'll understand Titans as a whole.