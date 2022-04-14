import os
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

metafile = 'data/tiktok_trainset.json'
embdir = '../../data/tiktok/embeddings/stam16_sbert_fc_weak_8x6x2_epoch4/'

def clean(vids):
    embs = []
    for vid in vids:
        emb = np.load(os.path.join(embdir, vid+'.npy'))
        embs.append(emb)
    
    embs = np.array(embs)
    cov_trace = np.trace(np.cov(embs.T))

    center = np.mean(embs, axis=0).reshape(1, -1)
    #print(center.shape)
    cossim = cosine_similarity(center, embs)[0]
    #print(cov_trace, cossim) 
    return cov_trace, cossim

if __name__ == '__main__':
    with open(metafile) as f:
        dataset = json.load(f)

    label2videos = {}
    for k in dataset.keys():
        for label in dataset[k]['label']:
            if label not in label2videos.keys():
                label2videos[label] = []
            label2videos[label].append(k)

    one_label = []
    for label in label2videos.keys():
        if len(label2videos[label]) <= 1:
            one_label.append(label)
    for label in one_label:
        del label2videos[label]

    newdataset = {}
    for label in tqdm.tqdm(label2videos.keys()):
        covtrace, sims = clean(label2videos[label])
        if covtrace < 0.6:
            for idx, sim in enumerate(sims):
                if sim > 0.7:
                    if label2videos[label][idx] not in newdataset:
                        newdataset[label2videos[label][idx]] = dataset[label2videos[label][idx]]
                        newdataset[label2videos[label][idx]]['label'] = [label]
                        del newdataset[label2videos[label][idx]]['rank']
                    else:
                        newdataset[label2videos[label][idx]]['label'].append(label)
    
    with open('data/tiktok_trainset_clean.json', 'w') as f:
        json.dump(newdataset, f, indent=4, separators=[',', ':'])