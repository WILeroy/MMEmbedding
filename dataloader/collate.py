import torch


def videoset_train_collate(batch):
    videos, vmasks, labels = [], [], []
    for item, label in batch:
        videos.append(item[0])
        vmasks.append(item[1])
        labels.append(label)
    return (torch.cat(videos, dim=0), torch.cat(vmasks, dim=0), 
            torch.cat(labels, dim=0))


def videoset_emb_collate(batch):
    video_batch, mask_batch, vid_batch = [], [], []
    for video, mask, vid in batch:
        video_batch.append(video)
        mask_batch.append(mask)
        vid_batch.append(vid)
    
    return (torch.cat(video_batch, dim=0), torch.cat(mask_batch, dim=0), 
            vid_batch)


def vaset_train_collate(batch):
    videos, vmasks, audios, amasks, labels = [], [], [], [], []
    for item, label in batch:
        videos.append(item[0])
        vmasks.append(item[1])
        audios.append(item[2])
        amasks.append(item[3])
        labels.append(label)
    return (torch.cat(videos, dim=0), torch.cat(vmasks, dim=0), 
            torch.cat(audios, dim=0), torch.cat(amasks, dim=0),
            torch.cat(labels, dim=0))


def vaset_emb_collate(batch):
    videos, vmasks, audios, amasks, vid_batch = [], [], [], [], []
    for video, vmask, audio, amask, vid in batch:
        videos.append(video)
        vmasks.append(vmask)
        audios.append(audio)
        amasks.append(amask)
        vid_batch.append(vid)
    return (torch.cat(videos, dim=0), torch.cat(vmasks, dim=0), 
            torch.cat(audios, dim=0), torch.cat(amasks, dim=0),
            vid_batch)

    
def vtset_train_collate(batch):
    videos, vmasks, texts, tmasks, labels = [], [], [], [], []
    for item in batch:
        videos.append(item[0])
        vmasks.append(item[1])
        texts.append(item[2])
        tmasks.append(item[3])
        labels.append(item[4])
    return (torch.cat(videos, dim=0), torch.cat(vmasks, dim=0), 
            torch.cat(texts, dim=0), torch.cat(tmasks, dim=0),
            torch.cat(labels, dim=0))


def vtset_emb_collate(batch):
    videos, vmasks, texts, tmasks, vid_batch = [], [], [], [], []
    for video, vmask, text, tmask, vid in batch:
        videos.append(video)
        vmasks.append(vmask)
        texts.append(text)
        tmasks.append(tmask)
        vid_batch.append(vid)
    return (torch.cat(videos, dim=0), torch.cat(vmasks, dim=0), 
            torch.cat(texts, dim=0), torch.cat(tmasks, dim=0),
            vid_batch)
