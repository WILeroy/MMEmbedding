import pandas as pd
import json
import os
import numpy as np
import utils.visulize as visulize
from sklearn.metrics.pairwise import cosine_similarity

embdir = '../../data/tiktok/embeddings/stam16_ss_16x6x2/'
metafile = 'data/tiktok_trainset.json'



def clean(vids):
    embs = []
    for vid in vids:
        emb = np.load(os.path.join(embdir, vid+'.npy'))
        embs.append(emb)
    
    embs = np.array(embs)
    cov_trace = np.trace(np.cov(embs.T))

    center = np.mean(embs, axis=0).reshape(1, -1)
    cossim = cosine_similarity(center, embs)[0].tolist()
    print(cossim)
    return cov_trace, cossim

def load_ranks(csv_path):
    df = pd.read_csv(csv_path)
    print(df.columns)

    contents = df["content"].to_list()
    keywords = df["keyword"].to_list()
    print(len(contents), len(keywords))
    keywords2rank = {}
    for idx, kword in enumerate(keywords):
        content = contents[idx]
        items = json.loads(content)
        keywords2rank[kword] = items

    return keywords2rank

if __name__ == '__main__':
    keyword2rank = load_ranks('../../data/tiktok/meta/tt_query_20220309_raw.csv')
    draw_data = []
    url_template = '../videos/{}.mp4'
    cnt = 0
    with open(metafile) as f:
        dataset = json.load(f)

    new_dataset = {}
    new_label = 0

    for kword in list(keyword2rank.keys())[:50]:
        qtext = kword
        gids, gurls, gtexts, greds, gauthors = [], [], [], [], []
        for video in keyword2rank[kword]:
            #print(video)
            if video['video_id'] in dataset.keys():
                gids.append(video['video_id'])
                gurls.append(url_template.format(video['video_id']))
                gtexts.append(video['caption'])
                gauthors.append(video['author_name'])
        covtrace, sims = clean(gids)

        if covtrace < 0.6:
            for idx, sim in enumerate(sims):
                if sim > 0.7:
                    greds.append(False)
                else:
                    greds.append(True)
        else:
            for idx, sim in enumerate(sims):
                greds.append(True)

        block_data = visulize.draw_text_to_videos(qtext+'</br>{:.5f}'.format(covtrace), gids, gurls, gtexts, gauthors, sims, greds)
        draw_data.append(block_data)
        if len(draw_data) == 5:
            visulize.draw_page(f'../../data/tiktok/visulize/{cnt}.html', draw_data, 10)
            cnt += 1
            draw_data = []

