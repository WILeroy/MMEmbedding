import argparse
import json
import os
import os.path as osp
from multiprocessing import Pool

import numpy as np
import tqdm

import utils.visulize as visulize
from utils.gallery import Gallery
from tools.build_allset import load_allset

videoid2url, _, _ = load_allset()

def run_top10(query_feat):
    global gallery
    
    y_score = gallery.cosines(query_feat)
    top10_indexes = np.argsort(-y_score)[1:10]
    top10_ids = [gallery.id(idx) for idx in top10_indexes]
    top10_scores = [y_score[idx] for idx in top10_indexes]

    return {'top10':top10_ids, 'top10_scores':top10_scores}


#################### global gallery ####################
gallery = Gallery()


def main(emb_dir, target_data):
    with open(target_data) as f:
        dataset = json.load(f)

    latestlabel = "-1"
    qids = []
    gids = []
    for k in dataset.keys():
        #if dataset[k]['label'] != latestlabel:
        qids.append(k)
        gids.append(k)
        latestlabel = dataset[k]['label']
        
    print('query num:', len(qids))
    print('gallery size:', len(gids))

    gallery.load_features(emb_dir, gids, True)

    pool = Pool(96)

    reqs = []
    for idx, qid in enumerate(qids):
        qemb = gallery._load_feat(osp.join(emb_dir, qid+'.npy'))
        reqs.append((idx, qid, pool.apply_async(run_top10, args=(qemb, ))))

    draw_data = []
    cnt = 0
    recall = 0
    for idx, qid, req in tqdm.tqdm(reqs):
        result = req.get()

        cosines = result['top10_scores']
        recall_ids = result['top10']
        recall_urls = []
        recall_captions = []
        recall_rights = []
        cntlocal = 0
        for rid in recall_ids:
            recall_urls.append(videoid2url[rid])
            recall_captions.append(dataset[rid]['caption'])
            if dataset[rid]['label'] == dataset[qid]['label']:
                recall_rights.append(True)
                cnt += 1
                cntlocal += 1
            else:
                recall_rights.append(False)

        recall += cntlocal * 1.0 / 9

        block_data = visulize.draw_video_to_videos(
            qid, videoid2url[qid], dataset[qid]['caption'], 
            recall_ids, cosines, recall_urls, recall_captions, recall_rights)

        draw_data.append(block_data)

    return cnt, recall / 399


if __name__ == '__main__':
    expname = 'stam16_sbert_fc_weak_8x6x2'
    for i in range(2, 17):
        cmd = f'python emb_stam_sbert_fc.py logs/{expname}/{expname}.json \
                logs/{expname}/model_{i}.pth data/testset400.json \
                ../../data/shopee-video/embeddings/{expname}/epoch_{i}'
        os.system(cmd)

        cnt, recall = main(f'../../data/shopee-video/embeddings/{expname}/epoch_{i}', 'data/testset400.json')
        print(f'epoch {i}:', cnt, recall)