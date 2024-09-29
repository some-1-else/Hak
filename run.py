from video_2 import extract_video_embedding_2, create_video_model
from VIT_2 import get_video_embedding, create_vit_model
from audio_2 import extract_audio_embeddings_from_video2, create_audio_model
from sklearn.metrics import f1_score
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import urllib

def check_dupl_from_db(vid_emb, vit_emb, aud_emb, database, thresholds_vid=(0.8, 0.95), thresholds_aud=(0.8, 0.95)):
    duplicates = []
    for uuid in database.keys():
        orig_vid_emb = np.array(database[uuid]['vid_emb'])
        orig_vit_emb = np.array(database[uuid]['vit_emb'])
        orig_aud_emb = np.array(database[uuid]['aud_emb'])
        if vid_emb is not None and orig_vid_emb is not None:
            try:
                similarity_vid = cosine_similarity(vid_emb.reshape(1, -1), orig_vid_emb.reshape(1, -1)).max(axis=1).mean()
            except:
                similarity_vid = None
        else:
            similarity_vid = None
        if vit_emb is not None and orig_vit_emb is not None:
            try:
                similarity_vit = cosine_similarity(vit_emb.reshape(1, -1), orig_vit_emb.reshape(1, -1)).max(axis=1).mean()
            except:
                similarity_vit = None
        else:
            similarity_vit = None            
        if aud_emb is not None and orig_aud_emb is not None:
            try:
                similarity_aud = cosine_similarity(aud_emb.reshape(1, -1), orig_aud_emb.reshape(1, -1)).max(axis=1).mean() 
            except:
                similarity_aud = None
        else:
            similarity_aud = None 
        similarity_vid = np.nanmean([similarity_vid, similarity_vit])
        if similarity_vid is None and similarity_aud is None:
            continue
        if similarity_aud is None:
            if similarity_vid > thresholds_vid[0]:
                duplicates.append((uuid, similarity_vid, similarity_aud))
                continue
        if similarity_vid is None:
            if similarity_aud > thresholds_aud[0]:
                duplicates.append((uuid, similarity_vid, similarity_aud))
                continue
        if similarity_vid > thresholds_vid[0] and similarity_aud > thresholds_aud[0]:
            duplicates.append((uuid, similarity_vid, similarity_aud))
        if similarity_vid > thresholds_vid[1] and similarity_aud < thresholds_aud[0]:
            duplicates.append((uuid, similarity_vid, similarity_aud))
            
    return duplicates

def find_duplicate_videos(video_file, database_path='database.json', model_name = 'resnet50', sample_frames=10, frame_size=(224, 224), threshold=0.9):
    database = json.load(open(database_path)) #БД с эмбеддингами
    #Модель для видеоряда
    model_vid = create_video_model()
    #model_vid = torch.nn.Sequential(*(list(model_vid.children())[:-1])) 
    #model_vid.eval()
    #Считаем метрики
    pred = []
    pred_uuid = []

    #Модель для аудиоряда
    model_aud = create_audio_model()
    model_vit = create_vit_model()


    

    updated = False

    #metadata = new_train#train.loc[train['uuid'].isin(['5eb4127e-5694-492b-963c-6688522e9ad2', '3726bb2d-3323-41f8-8eb2-0d7cf095d62b'])].sort_values('created')
    #Перебираем все файлы с видео
    #for video_file in tqdm(video_files, desc="Extracting video features"):
    if updated:
        database = json.load(open(database_path)) #БД с эмбеддингами
        updated = False
    #audio_embeddings = {}
    #duplicates = []
    #video_file = os.path.join(video_folder, row.link.split('/')[-1])
    # Путь для временного хранения аудио
    #temp_audio_path = os.path.join(f"{video_file}.wav")
    #Получаем аудио эмбеддинг
    

    audio_embedding = extract_audio_embeddings_from_video2(video_file, model_aud)
    #Получаем видео эмбеддинг
    video_features = extract_video_embedding_2(video_file, model_vid)
    vit_embedding = get_video_embedding(video_file, model_vit)
    #Ищем оригинал в БД
    duplicates = check_dupl_from_db(video_features, vit_embedding, audio_embedding, database=database)
    #Если есть выводим, иначе записываем новый оригинал в БД
    if duplicates:
        duplicates.sort(key=lambda x: np.mean([x[1], x[2]]))
        pred.append(1)
        pred_uuid.append(duplicates[-1][0])
        #print(f'Дубликат: {video_file}, оригинал: {duplicates}') 
    else:
        pred.append(0)
        pred_uuid.append(np.nan)
        if video_features is not None:
            vid_emb = video_features.tolist()
        else:
            vid_emb = None
        if vit_embedding is not None:
            vit_emb = vit_embedding.tolist()
        else:
            vid_emb = None                
        if audio_embedding is not None:
            aud_emb = audio_embedding.tolist()
        else:
            aud_emb = None
        new_origin = {video_file.split('/')[-1]: {'vid_emb': vid_emb,
                                 'vit_emb': vit_emb,
                                 'aud_emb': aud_emb}}
        database.update(new_origin)
        json.dump(database, open(database_path, 'w'))
        updated = True
        #print('Добавили в БД:', row.uuid)
                

            
       # if os.path.exists(temp_audio_path):
            #os.remove(temp_audio_path)
    return pred, pred_uuid

def run(path2video):
    os.chdir('/home/user1')
   # path2video = os.path.join('Hakaton/models/test', url_link.split('/')[-1])
    #urllib.request.urlretrieve(url_link, path2video) 
    pred, pred_uuid = find_duplicate_videos(path2video, database_path='Hakaton/models/database.json')
    return pred, pred_uuid
    