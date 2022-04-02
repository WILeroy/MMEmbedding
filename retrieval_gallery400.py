import argparse
import json
import os.path as osp
from multiprocessing import Pool

import numpy as np
import tqdm

import utils.visulize as visulize
from utils.gallery import Gallery
from tools.build_allset import load_allset


def run_top10(query_feat):
    global gallery
    
    y_score = gallery.cosines(query_feat)
    top10_indexes = np.argsort(-y_score)[1:10]
    top10_ids = [gallery.id(idx) for idx in top10_indexes]
    top10_scores = [y_score[idx] for idx in top10_indexes]

    return {'top10':top10_ids, 'top10_scores':top10_scores}


#################### global gallery ####################
gallery = Gallery()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_dir')
    parser.add_argument('target_data')
    parser.add_argument('--save-gallery', type=str, default=None)
    parser.add_argument('--reload', type=str, default=None)
    parser.add_argument('--save-result', type=str, default='shopee-video_results.html')
    args = parser.parse_args()

    videoid2url, _, _ = load_allset()

    with open(args.target_data) as f:
        dataset = json.load(f)

    latestlabel = "-1"
    qids = []
    gids = []
    for k in dataset.keys():
        if dataset[k]['label'] != latestlabel:
            qids.append(k)
        gids.append(k)
        latestlabel = dataset[k]['label']
        
    print('query num:', len(qids))
    print('gallery size:', len(gids))

    if args.reload is not None:
        gallery.reload(args.reload, True)
    else:
        gallery.load_features(args.emb_dir, gids, True)

    if args.save_gallery is not None:
        gallery.save(args.save_gallery)

    pool = Pool(96)

    reqs = []
    for idx, qid in enumerate(qids):
        qemb = gallery._load_feat(osp.join(args.emb_dir, qid+'.npy'))
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
    visulize.draw_page(args.save_result, draw_data, 5)

    print('right num:', cnt)
    print('recall:', recall / 40)
