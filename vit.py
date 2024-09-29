import os
import cv2
import torch
import numpy as np
from PIL import Image  # Импортируем Image из PIL
from torchvision import transforms
from torchvision.models import vit_b_16  # Импортируем предобученный ViT
from scipy.spatial.distance import cosine

# Загрузка предобученной модели Vision Transformer (ViT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit_b_16(weights='DEFAULT')  # Загрузка предобученных весов
model = model.to(device)
model.eval()

# Преобразования для подготовки видео с аугментацией
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Размер, ожидаемый моделью
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Изменение яркости, контрастности и цветового оттенка
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_frames(video_path, num_frames=10):
    """ Извлекает ключевые кадры из видео """
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
    return frames

def get_video_features(video_path, num_frames=10):
    """ Извлекает признаки из видео с помощью ViT """
    frames = extract_frames(video_path, num_frames)
    if not frames:
        return None
    
    # Применяем преобразования и извлекаем признаки
    features_list = []
    for frame in frames:
        # Применяем аугментацию и преобразования
        transformed_frame = transform(frame).unsqueeze(0).to(device)  # Добавляем размерность батча
        with torch.no_grad():  # Отключаем отслеживание градиентов
            features = model(transformed_frame).detach().cpu().numpy().flatten()  # Преобразуем в NumPy
        features_list.append(features)

    # Объединяем признаки всех кадров в один вектор
    return np.mean(features_list, axis=0)  # Средние признаки всех кадров

def cosine_similarity(vec1, vec2):
    """ Рассчитывает косинусное расстояние между двумя векторами """
    if vec1 is None or vec2 is None:
        return float('inf')  # Если одно из значений отсутствует
    return cosine(vec1, vec2)

def find_duplicates(video_dir, threshold=0.3, num_frames=10):
    """ Ищет дубликаты видео в указанной директории по косинусному расстоянию """
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
    video_features = {}

    # Извлекаем признаки для каждого видео
    for video_file in video_files:
        features = get_video_features(video_file, num_frames)
        video_features[video_file] = features

    duplicates = []

    # Сравниваем признаки между видео
    for i, video_1 in enumerate(video_files):
        for j, video_2 in enumerate(video_files):
            if i >= j:
                continue

            distance = cosine_similarity(video_features[video_1], video_features[video_2])

            # Проверяем на дубликаты с учетом порога
            if distance < threshold:
                duplicates.append({
                    'video_1': video_1,
                    'video_2': video_2,
                    'cosine_distance': distance,
                })

    return duplicates

# video_dir = 'video2' 
# threshold_value = 0.1  # Чем меньше - тем точнее!!!!
# num_frames_to_analyze = 10
# duplicates = find_duplicates(video_dir, threshold=threshold_value, num_frames=num_frames_to_analyze)

# # Выводим только дубликаты
# if duplicates:
#     for dup in duplicates:
#         print(f"Дубликаты найдены: {dup['video_1']} и {dup['video_2']}")
#         print(f"Косинусное расстояние: {dup['cosine_distance']}\n")
# else:
#     print("Дубликаты не найдены.")