from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse
from fastapi_sqlalchemy import db
from pydantic import Field
import torch
import numpy as np
import os
from scipy.spatial.distance import cosine
from torch.nn.functional import cosine_similarity
import shutil
import io

with open("embedings/1_avi_video.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())
    print(buffer)
emb = torch.load(buffer, map_location=torch.device('cpu'), weights_only=False)
print(emb)
