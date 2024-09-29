#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

def create_vit_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Загрузка предобученной модели Vision Transformer (ViT)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_b_16(weights='DEFAULT')
    model = model.to(device)
    model.eval()
    return model

def get_video_embedding(video_path, model, num_frames=10):
    """ Извлекает эмбеддинг видеофайла с помощью ViT """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Преобразования для подготовки видео
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        # Извлечение кадров из видео
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Конвертируем в RGB
                frames.append(Image.fromarray(frame))  # Конвертируем в PIL.Image

        cap.release()

        if not frames:
            return None
        
        # Извлечение признаков из кадров
        features_list = []
        for frame in frames:
            transformed_frame = transform(frame).unsqueeze(0).to(device)  # Добавляем размерность батча
            with torch.no_grad():
                features = model(transformed_frame).detach().cpu().numpy().flatten()
            features_list.append(features)

        #return np.mean(features_list, axis=0)  # Возвращаем средние признаки всех кадров
        return np.array(features_list)
    
    except Exception as e:
        print(f"Ошибка при извлечении эмбеддингов из {video_path}: {e}")
        return None

