import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def extract_video_features(video_path, sample_frames=10, frame_size=(224, 224), model_name = "resnet50"):
    model = getattr(models, model_name)(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1])) 
    model.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(frame_count // sample_frames, 1)
    
    features = []
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, frame_size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb).unsqueeze(0)
        
        with torch.no_grad():
            feature = model(frame_tensor).squeeze()
        
        features.append(feature)
    
    cap.release()

    # return features
    return torch.stack(features).mean(dim=0) if features else None

def find_duplicate_videos(directory, model_name = 'resnet50', sample_frames=10, frame_size=(224, 224), threshold=0.9):

    # model = getattr(models, model_name)(pretrained=True)
    # model = torch.nn.Sequential(*(list(model.children())[:-1])) 
    # model.eval()
    
    # Define transform
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(frame_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    video_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    video_features = {}
    
    for video_file in tqdm(video_files, desc="Extracting video features"):
        print(video_file)
        video_features[video_file] = extract_video_features(video_file, sample_frames, frame_size)
    
    duplicates = []
    compared_videos = set()
    video_files_sorted = sorted(video_features.keys())
    
    for i, video1 in enumerate(video_files_sorted):
        for j, video2 in enumerate(video_files_sorted[i + 1:], i + 1):
            if (video1, video2) in compared_videos or (video2, video1) in compared_videos:
                continue
            if video_features[video1] is None or video_features[video2] is None:
                continue
            similarity = cosine_similarity(video_features[video1].unsqueeze(0), video_features[video2].unsqueeze(0)).item()
            if similarity > threshold:
                duplicates.append((video1, video2, similarity))
            compared_videos.add((video1, video2))
    
    return duplicates

# video_directory = "video"

# duplicates = find_duplicate_videos(
#     directory = video_directory,
#     model_name ='resnet50',  # 'resnet152' 'resnet101'
#     sample_frames = 5,
#     frame_size =(224, 224),
#     threshold = 0.9
# )

# # Output the duplicates
# if duplicates:
#     print("Found duplicate videos:")
#     for dup in duplicates:
#         print(f"{dup[0]} {dup[1]} | {dup[2]:.4f}")
