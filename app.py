import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

from streamlit_echarts import st_echarts
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
st.set_page_config(layout="wide")

import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch
from torch import nn
from typing import List, Callable, Optional
import os

import pandas as pd


labels = ["adenocarcinoma","large.cell","normal","squamous.cell"]
model_name_or_path = 'alicelouis/ViTLungMiNi'

@st.cache_resource(show_spinner=False,ttl=1800,max_entries=2)
def FeatureExtractor(model_name_or_path):
    feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
    return feature_extractor


@st.cache_resource(show_spinner=False,ttl=1800,max_entries=2)
def LoadModel(model_name_or_path):
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={int(i): c for i, c in enumerate(labels)},
        label2id={c: int(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True)
    return model


# Model wrapper to return a tensor
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

# """ Translate the category name to the category index.
#     Some models aren't trained on Imagenet but on even larger datasets,
#     so we can't just assume that 761 will always be remote-control.

# """
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
    
# """ Helper function to run GradCAM on an image and create a visualization.
#     (note to myself: this is probably useful enough to move into the package)
#     If several targets are passed in targets_for_gradcam,
#     e.g different categories,
#     a visualization for each of them will be created.
    
# """
 
def print_top_categories(model, img_tensor, top_k=5):
    feature_extractor = FeatureExtractor(model_name_or_path)
    inputs = feature_extractor(images=img_tensor, return_tensors="pt")
    outputs = model(**inputs)
    logits  = outputs.logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
    probabilities = nn.functional.softmax(logits, dim=-1)
    topK = dict()
    for i in indices:
        topK[model.config.id2label[i]] = probabilities[0][i].item()*100
    return topK

def reshape_transform_vit_huggingface(x):
    activations = x[:, 1:, :]
    
    activations = activations.view(activations.shape[0],
                                   14, 14, activations.shape[2])
    
    activations = activations.transpose(2, 3).transpose(1, 2)
    
    return activations




st.header('Lung Cancer classification with Vision Transformer: จำแนกมะเร็งปอด')
with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
# end def
with st.sidebar:
    tabs = on_hover_tabs(tabName=['Model', 'Money', 'Economy'], 
                         iconName=['dashboard', 'money', 'economy'], default_choice=0)




# with open("assets/css/style.css") as f:
#     st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
# hide_table_index = """
#             <style>         
#             thead {display:none}  
#             tbody th {display:none}
#             .blank {display:none}
#             </style>
#             """ 
# st.markdown(hide_table_index, unsafe_allow_html=True)


if tabs =='Model':
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ")

    if uploaded_file is not None:
        model_name_or_path = 'alicelouis/ViTLungMiNi'
        feature_extractor = FeatureExtractor(model_name_or_path)
        model = LoadModel(model_name_or_path)
        img = Image.open(uploaded_file)
        img_out = img.resize((224,224))
        img_out = np.array(img_out)
        # โหลดโมเดลที่เซฟ
        image = img.resize((224,224))
        img_tensor = transforms.ToTensor()(image)
        def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module=img_tensor,
                          input_image: Image=image,
                          method: Callable=GradCAM):
            with method(model=HuggingfaceToTensorModelWrapper(model),
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
                repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

                batch_results = cam(input_tensor=repeated_tensor,
                                    targets=targets_for_gradcam)
                results = []
                for grayscale_cam in batch_results:
                    visualization = show_cam_on_image(np.float32(input_image)/255,
                                                    grayscale_cam,
                                                    use_rgb=True)
                    # Make it weight less in the notebook:
                    visualization = cv2.resize(visualization,
                                            (visualization.shape[1]//2, visualization.shape[0]//2))
                    results.append(visualization)
                return np.hstack(results)
                
        inputs = feature_extractor(images=image, return_tensors="pt")
        targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, "adenocarcinoma")),
                       ClassifierOutputTarget(category_name_to_index(model, "large.cell")),
                       ClassifierOutputTarget(category_name_to_index(model, "normal")),
                       ClassifierOutputTarget(category_name_to_index(model, "squamous.cell"))
                       ]
        target_layer_dff = model.vit.layernorm
        target_layer_gradcam = model.vit.encoder.layer[-2].output
        image_resized = image.resize((224, 224))
        tensor_resized = transforms.ToTensor()(image_resized)
        outputs = model(**inputs)
        logits = outputs.logits
        # model predicts one of the 4 classes
        predicted_class_idx = logits.argmax(-1).item()
        className = labels[predicted_class_idx]
        st.success("Predicted class is : " + className , icon="✅")
        st.image(Image.fromarray(run_dff_on_image(model=model,
                          target_layer=target_layer_dff,
                          classifier=model.classifier,
                          img_pil=image_resized,
                          img_tensor=tensor_resized,
                          reshape_transform=reshape_transform_vit_huggingface,
                          n_components=4,
                          top_k=4)))
        st.image(Image.fromarray(run_grad_cam_on_image(model=model,
                      target_layer=target_layer_gradcam,
                      targets_for_gradcam=targets_for_gradcam,
                      input_tensor=tensor_resized,
                      input_image=image_resized,
                      reshape_transform=reshape_transform_vit_huggingface)))
        topK = print_top_categories(model, tensor_resized)
        df = pd.DataFrame.from_dict(topK, orient='index')
        st.dataframe(df)
        list_to_be_sorted= []
        for x, y in topK.items():
            dic = dict()
            dic["value"] = y
            dic["name"] = x
            print(x, y)
            list_to_be_sorted.append(dic)
        
        print(list_to_be_sorted)

        # list_to_be_sorted= [
        # { "value": 335, "name": 'Direct' },
        # { "value": 310, "name": 'Email' },
        # { "value": 274, "name": 'Union Ads' },
        # { "value": 235, "name": 'Video Ads' },
        # { "value": 400, "name": 'Search Engine' }]
        newlist = sorted(list_to_be_sorted, key=lambda d: d['value']) 
        options = {
  "backgroundColor": '#2c343c',
  "title": {"text": 'Customized Pie',"left": 'center',"top": 5
            ,"textStyle": {"color": '#ccc'}},
  "tooltip": {"trigger": 'item'},
  "visualMap": {
    "show": False,
    "min": 0,
    "max": 150,
    "inRange": {
      "colorLightness": [0, 1]
    }
  },
  "series": [
    {
      "name": 'Access From',
      "type": 'pie',
      "radius": '70%',
      "center": ['50%', '50%'],
      "data":newlist,
      "roseType": 'radius',
      "label": {
        "color": 'rgba(255, 255, 255, 0.3)'
      },
      "labelLine": {
        "lineStyle": {
          "color": 'rgba(255, 255, 255, 0.3)'
        },
        "smooth": 0.2,
        "length": 10,
        "length2": 60
      },
      "itemStyle": {
        "color": '#c23531',
        "shadowBlur": 200,
        "shadowColor": 'rgba(0, 0, 0, 0.5)'
      },
      "animationType": 'scale',
      "animationEasing": 'elasticOut',
      
    }
  ]
}
        st_echarts(options=options)

        st.balloons()


                
            
            
        