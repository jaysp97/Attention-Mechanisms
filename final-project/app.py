# app.py

from flask import Flask, request, render_template, redirect, url_for
import os
import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import skimage.io as io
import PIL.Image

from lib.CLIP_prefix_caption.predict import MLP, ClipCaptionModel, generate2, generate_beam

import logging

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

WEIGHTS_PATHS = {
    "coco": "coco_weights.pt",
    "conceptual-captions": "conceptual_weights.pt",
}

D = torch.device
CPU = torch.device("cpu")

current_directory = os.getcwd()
save_path = os.path.join('',"lib/CLIP_prefix_caption/pretrained_models")
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'coco_weights.pt')

is_gpu = torch.cuda.is_available()
device = "cuda" if is_gpu else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prefix_length = 10

model = ClipCaptionModel(prefix_length)

model.load_state_dict(torch.load(model_path, map_location=CPU))

model = model.eval() 
model = model.to(device)

def get_caption(filename, use_beam_search=True):
    # logging.info(filename+'\n')
    print(filename)
    image = io.imread(filename)
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text_prefix

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('uploaded_file', filename=filename))

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(get_caption(filepath))
    caption1 = get_caption(filepath, use_beam_search=True)
    caption2 = get_caption(filepath, use_beam_search=False)
    return render_template('display_image.html', filename=filename, custom_string1 = caption1, custom_string2= caption2)

if __name__ == '__main__':
    app.run(debug=True)
    