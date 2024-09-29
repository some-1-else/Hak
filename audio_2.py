#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pydub import AudioSegment
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch

def create_audio_model():
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

def extract_audio_embeddings_from_video2(video_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Загружаем аудио из видеофайла
        audio = AudioSegment.from_file(video_path)
        audio_array = np.array(audio.get_array_of_samples())
        
        # Частота дискретизации - 16 кГц
        sr = 16000
        y = audio_array.astype(np.float32)

        # Аугментация аудио
        noise = np.random.randn(len(y))
        y_noise = y + 0.005 * noise
        shift = np.roll(y, sr // 10)
        y_stretch = librosa.effects.time_stretch(y, rate=0.8)
        y_augmented = librosa.effects.preemphasis(y_stretch)

        augmented_audios = [y_noise, shift, y_stretch, y_augmented]
        
        embeddings = []
        for augmented_audio in [y] + augmented_audios:
            waveform = tf.convert_to_tensor(augmented_audio, dtype=tf.float32)
            waveform = tf.reshape(waveform, [-1])  
            scores, embedding, _ = model(waveform) 

            embeddings.append(tf.reduce_mean(embedding, axis=0).numpy())
        
        embeddings = np.array(embeddings)
        
        return embeddings

    except Exception as e:
        print(f"Ошибка при извлечении эмбеддингов из {video_path}: {e}")
        return None

