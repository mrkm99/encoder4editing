from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
 
from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.
import dlib
from utils.alignment import align_face
import tempfile,glob


import streamlit as st
# モデルに学習済みパラメータをロード
@st.cache
def load_model(model_path='pretrained_models/e4e_ffhq_encode.pt'):
  ckpt = torch.load(model_path, map_location='cpu')
  opts = ckpt['opts']
  opts['checkpoint_path'] = model_path
  opts= Namespace(**opts)
  net = pSp(opts)
  net.eval()
  net.cuda()
  return net

@st.cache
def run_alignment(image_path):
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  return aligned_image 

def setting_attribute_slider(attr_path = "./editings/interfacegan_directions/"):
  attrs = ["age","pose","smile"]
  attr_params = {}
  for attr in attrs:
    attr_param = torch.load(attr_path+attr+".pt")
    attr_params[attr] = st.sidebar.slider(attr,-3.0,3.0,0.0,0.1)*attr_param.to("cuda")
  return attr_params

img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

net = load_model()

f = st.sidebar.file_uploader("input image")

attr_params = setting_attribute_slider()

cols = st.beta_columns(2)
if not f == None:
  tfile = tempfile.NamedTemporaryFile(delete=False)
  tfile.write(f.read())
  image = run_alignment(tfile.name)

  if not image == None:
    with torch.no_grad():
      transformed_image = img_transforms(image)
      images, latents = net(transformed_image.unsqueeze(0).to('cuda').float(), randomize_noise=False, return_latents=True)
      result_image, latent = images[0], latents[0]
      with cols[0]:
        st.write("input image")
        st.image(image)
      with cols[1]:
        st.write("inversion image")
        st.image(tensor2im(result_image))
      edit_latent = latent
      for attr in attr_params.keys():
        edit_latent += attr_params[attr][0]

      generator = net.decoder
      edit_images,_ = generator([edit_latent.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)

    st.image(tensor2im(edit_images[0]))