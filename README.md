# MMSimilarity

## Model

* video experts
    * stam

* audio experts
    * vggish

* text experts
    * sentence-transformer

## Train 

### Self-Supervised

### Supervised

* Download pretrained model

```
mkdir pretrain
cd pretrain
wget https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/STAM/v2/stam_16.pth
```

* preprocess dataset

* train

```
python train_stam_sbert_weak.py config/*.json
```

## Extract