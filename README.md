# ANDI


This is implementation of ANDI: a Joint Disambiguation Framework Integrating Author Name Disambiguation Goals.

In the competition [WhoIsWho:Name Disambiguation from Scratch](https://www.biendata.xyz/competition/whoiswho1/final-leaderboard/) our method got the best results!


![Framework](/ANDI.png)

## Datasets
Aminer-v3: https://www.aminer.cn/whoiswho

Aminer-na: https://static.aminer.cn/misc/na-data-kdd18.zip

Citeseer: https://clgiles.ist.psu.edu/data/


## Requirements
Python 3.8

    gensim==4.3.2
    loguru==0.7.0
    numpy==1.24.4
    pinyin==0.4.0
    scikit_learn==1.3.0
    torch==2.0.1
    tqdm==4.66.1
    transformers==4.31.0

## Usage
Download [dataset(OneDriver)](https://stuxmueducn-my.sharepoint.com/:u:/g/personal/liutao2676_stu_xmu_edu_cn/EZdjTOlPfjZBhhFJVbqj524BYtp0Z-IMYk13OQGAEd-FOA?e=wAcl9b), unzip the file and put the data directory into project directory, project organize as follows:

    .
    ├── ANDI/
    ├── dataset/
        ├── Aminer-na/
        │   ├── processed_data/
        │   │   └── rel-embs/
        │   ├── valid_author.json
        │   ├── valid_pub.json
        │   ├── train_author.json
        │   └── train_pub.json
        ├── Aminer-v3/
        │   ├── processed_data/
        │   ├── test_author.json
        │   ├── test_pub.json
        │   ├── valid_author.json
        │   ├── valid_pub.json
        │   ├── train_author.json
        │   └── train_pub.json
        ├── CiteSeerX/
        │   ├── train_author.json
        │   └── train_pub.json
        └── data/
            ├── gene/
            │   ├── Aminer-na/
            │   ├── Aminer-v3/
            │   └── CiteSeerX/
            ├── na/
            ├── v3/
            └── x/

data processing:

    data_loader.py

train:

    sem_train.py
    rel_train.py

test:

    snd_train_Aminer-v3.py
    snd_train_Aminer-na.py
    snd_train_CiteSeerX.py

