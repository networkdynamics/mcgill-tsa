# McGill Twitter Sentiment Analysis dataset (MTSA)
This repository stores all code and files related to the MTSA dataset. The corresponding paper accepted at NAACL 2018, "Sentiment Analysis: It's Complicated", can be found online [here](aclweb.org/anthology/N18-1171).

**You may only use this code and dataset for _non-proprietary research_ purposes, as per Twitter's terms of service.** Additionally, note that this project uses the GPU GPLv3 license, meaning that you may not incorporate this program "into proprietary programs".

## Overview of provided code and data
The provided Python code is written and designed for `Python 3.6.x`.

### Package dependencies (mainly for preprocessing):
* numpy
* nltk
* autocorrect [(repository)](https://github.com/phatpiglet/autocorrect/)
* tweet-preprocessor [(repository)](https://pypi.python.org/pypi/tweet-preprocessor/0.4.0)
* wordninja [(repository)](https://github.com/keredson/wordninja)
* progress (bar) [(repository)](https://pypi.python.org/pypi/progress)

### Data files
In the directory *data*:
```
data/annotated_tweets.csv            ==> the annotated tweets as given by CrowdFlower
data/unannotated_tweets.csv          ==> the unannotated tweets before being sent to CrowdFlower
data/processed_annotated_tweets.npy  ==> numpy pickle file resulting from after data is preprocessed
```

### Code files
In the directory *src*:
```
src/load_tweets.py    ==> main file, loads all tweets from crowdflower csv
src/preprocessing.py  ==> code for preprocessing tweets
src/tweet.py          ==> handy tweet objects, contains original and preprocessed text,
                          amenable to adding specific feature sets
```

Code used for feature extraction and experimental design in the paper is available on request.

## Contact
Contact [Kian Kenyon-Dean](https://kiankd.github.io/) at *kian.kenyon-dean@mail.mcgill.ca* (or, [on github](https://github.com/kiankd))  for questions about this repository.
