import os
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import pkg_resources
pkg_resources.require("numpy==1.26.4")
import numpy as np

# моделька
model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_audio_from_video(video_path, output_audio_path):

    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(output_audio_path, format="wav")
        return True
    except Exception as e:
        print(f"Ошибка при извлечении аудио из {video_path}: {e}")
        return False

def augment_audio(y, sr):
    
    # белый шум
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise

    # сдвиг по времени
    shift = np.roll(y, sr // 10)

    # растяжение времени
    y_stretch = librosa.effects.time_stretch(y, rate=0.8)

    # громкость
    y_augmented = librosa.effects.preemphasis(y_stretch)
    
    return [y_noise, shift, y_stretch, y_augmented]

def extract_audio_embeddings(audio_path):

    try:
        # частота дискретизации - 16 кГц
        y, sr = librosa.load(audio_path, sr=16000)
        y = y.astype(np.float32)

        augmented_audios = augment_audio(y, sr)
        
        embeddings = []
        for augmented_audio in [y] + augmented_audios:
           
            waveform = tf.convert_to_tensor(augmented_audio, dtype=tf.float32)
          
            waveform = tf.reshape(waveform, [-1])  
            scores, embedding, _ = model(waveform) 

            embeddings.append(tf.reduce_mean(embedding, axis=0).numpy())
        
        embeddings = np.mean(embeddings, axis=0)
        
        return embeddings
    except Exception as e:
        print(f"Ошибка при извлечении эмбеддингов из {audio_path}: {e}")
        return None

def find_duplicate_videos(video_folder, similarity_threshold=0.8):

    audio_embeddings = {}
    duplicates = []
    
    # Перебираем все файлы в папке
    for file_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, file_name)
        
        if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue 
        
        # Путь для временного хранения аудио
        temp_audio_path = os.path.join(video_folder, f"{file_name}.wav")
        
        if extract_audio_from_video(video_path, temp_audio_path):
            audio_embedding = extract_audio_embeddings(temp_audio_path)
            
            if audio_embedding is not None:
                for existing_video, existing_embedding in audio_embeddings.items():
                    similarity = 1 - cosine(audio_embedding, existing_embedding)
                    if similarity > similarity_threshold:
                        duplicates.append((video_path, existing_video, similarity))
                        break
                
                audio_embeddings[video_path] = audio_embedding
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    return duplicates

def find_duplicates_in_folder(video_folder, similarity_threshold=0.8):
    
    duplicates = find_duplicate_videos(video_folder, similarity_threshold=similarity_threshold)
    
    if duplicates:
        for duplicate, original, similarity in duplicates:
            print(f"{duplicate} {original} | {similarity:.2f}")

# video_folder = 'video'
# similarity_threshold = 0.9
# find_duplicates_in_folder(video_folder, similarity_threshold)
