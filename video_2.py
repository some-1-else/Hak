#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
def create_video_model(model_name='resnet50', use_pretrained=True):
    # Загрузка предобученной модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, model_name)(pretrained=use_pretrained)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Удаляем последний слой
    model.eval()
    model.to(device)
    return model
    
def extract_video_embedding_2(video_path, 
                              model,
                            sample_frames=10, 
                            frame_size=(224, 224), 
                            
                            ):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Определяем трансформации
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = frame_count // sample_frames

    features = []
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Преобразуем и нормализуем кадр
        frame = cv2.resize(frame, frame_size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feature = model(frame_tensor).squeeze()
        
        features.append(feature)
        if len(features) == 10:
            break
    
    cap.release()
    
    # Возвращаем усредненный вектор признаков в формате np.ndarray
    #return torch.stack(features).mean(dim=0).numpy() if features else None
    emb = torch.stack(features).cpu().numpy()
    #print('Vid:', emb.shape)
    return emb if features else None #возвращаем набор эмбеддингов для всех кадров

