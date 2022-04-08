# MMSimilarity

## Summary

| video experts | audio experts | text experts | fusions |
| :-:| :-: | :-: | :-: |
| STAM | VGGish | Sentence-Transformer | MLP |

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

```
# download meta file, url at data/metafile_url
gdwon --fuzzy [metafile url]

# modify build_allset.py
python build_allset.py

# modify build_testset400.py
python build_testset400.py

# modify build_weak_trainset.py
python build_weak_trainset.py
```

* train

```
python train_stam_sbert_weak.py config/*.json
```

## Extract
```
python emb_stam_sbert_fc.py logs/*/*.json logs/*/model_*.pth data/testset400.json [dir_to_save_embeddings]
```

## Retrieval

```
python retrieval_gallery400.py [emb_dir] data/testset400.json
```