import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import PIL.Image as Image
import csv

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
st.markdown('''
<style>
    section[data-testid='stSidebar'] {
        background-color: #111;
        min-width: unset !important;
        width: unset !important;
        flex-shrink: unset !important;
    }

    button[kind="header"] {
        background-color: transparent;
        color: rgb(180, 167, 141);
    }

    @media (hover) {
        /* header element to be removed */
        header["data"-testid="stHeader"] {
            display: none;
        }

        /* The navigation menu specs and size */
        section[data-testid='stSidebar'] > div {
            height: 100%;
            width: 95px;
            position: relative;
            z-index: 1;
            top: 0;
            left: 0;
            background-color: #111;
            overflow-x: hidden;
            transition: 0.5s ease;
            padding-top: 60px;
            white-space: nowrap;
        }

        /* The navigation menu open and close on hover and size */
        /* section[data-testid='stSidebar'] > div {
        height: 100%;
        width: 75px; /* Put some width to hover on. */
        /* } 

        /* ON HOVER */
        section[data-testid='stSidebar'] > div:hover{
        width: 300px;
        }

        /* The button on the streamlit navigation menu - hidden */
        button[kind="header"] {
            display: none;
        }
    }

    @media (max-width: 272px) {
        section["data"-testid='stSidebar'] > div {
            width: 15rem;
        }/.
    }
</style>
''', unsafe_allow_html=True)

@st.cache(allow_output_mutation=True,show_spinner=False,ttl=1800,max_entries=2,persist=True)
def FeatureExtractor(model_name_or_path):
    feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
    return feature_extractor


@st.cache(allow_output_mutation=True,show_spinner=False,ttl=1800,max_entries=2,persist=True)
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
#     Some models aren't trained on Imagenet but on even larger "data"sets,
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


def count_system():
    with open('system.txt', 'r') as f:
    # Count the number of non-blank lines in the file
        count = sum(1 for line in f if line.strip())
    # Print the count
    f.close()
    return int(count)

st.markdown(
    """
<div style='border: 2px solid green; border-radius: 5px; padding: 10px; background-color: white;'>
    <h1 style='text-align: center; color: green;'>🏥 Lung Cancer Classification with Vision Transformer : จำแนกมะเร็งปอด 🫁</h1>
</div>
    """, unsafe_allow_html=True)

with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
# end def
with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home','Upload', 'Analytics', 'More Information'], iconName=['home','upload', 'analytics', 'informations'], styles={'navtab': {'background-color': '#111', 'color': '#818181', 'font-size': '18px', 'transition': '.3s', 'white-space': 'nowrap', 'text-transform': 'uppercase'}, 'tabOptionsStyle': {':hover :hover': {'color': 'red', 'cursor': 'pointer'}}, 'iconStyle': {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'}, 'tabStyle': {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'}}, key="1",default_choice=0)
    st.markdown(
    """
        <div style='border: 2px solid green; padding: 10px; white; margin-top: 5px; margin-buttom: 5px; margin-right: 20px; bottom: 50;'>
            <h1 style='text-align: center; color: green; font-size: 100%'> ได้รับทุนสนับสนุน 2,000 บาท </h1>
            <h1 style='text-align: center; color: green; font-size: 100%'> National Software Contest ครั้งที่ 25 </h1>
            <h1 style='text-align: center; color: green; font-size: 100%'> ประจำปีงบประมาณ 2566 </h1>
        </div>
    """, unsafe_allow_html=True)

data_base = []

if tabs == 'Home':
    st.image('How_to_use.png',use_column_width=True)

elif tabs == 'Upload' and count_system () != 1:
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ")
    if uploaded_file is not None:
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
            # display the images on streamlit
            dff_image = Image.fromarray(run_dff_on_image(model=model,
                                        target_layer=target_layer_dff,
                                        classifier=model.classifier,
                                        img_pil=image_resized,
                                        img_tensor=tensor_resized,
                                        reshape_transform=reshape_transform_vit_huggingface,
                                        n_components=4,
                                        top_k=4))
            gradcam_image = Image.fromarray(run_grad_cam_on_image(model=model,
                                    target_layer=target_layer_gradcam,
                                    targets_for_gradcam=targets_for_gradcam,
                                    input_tensor=tensor_resized,
                                    input_image=image_resized,
                                    reshape_transform=reshape_transform_vit_huggingface))
            # save the images to disk
            dff_image.save(r".\save_images\dff_image.png")
            gradcam_image.save(r".\save_images\gradcam_image.png")
            topK = print_top_categories(model, tensor_resized)
            df = pd.DataFrame.from_dict(topK, orient='index')
            list_to_be_sorted= []
            for x, y in topK.items():
                dic = dict()
                dic["value"] = y
                dic["name"] = x
                list_to_be_sorted.append(dic)
                data_base.append(y)
            with open('system.txt', 'a') as f:
                    f.write(str(topK) + '\n')
            f.close()       
            out_load = []
            with open('className.txt', 'a') as f:
                for value in className:
                    f.write(str(value))
            f.close()  
            out_load = []
            with open('className.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out_load.append(line)
            f.close()  
            if out_load[0] == 'adenocarcinoma':
                st.markdown(
                        """
                    <div style='border: 2px solid #5370c6; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: #5370c6; font-size: 180%'> Adenocarcinoma </h3>
                    </div>
                        """, unsafe_allow_html=True)   
                st.markdown(
                            """
                        <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h4 style='text-align: center; color: black; font-size: 150%'> พบประมาณ 40% ของคนที่เป็นมะเร็งปอดทั้งหมดที่เป็นผู้ชายและผู้หญิง แม้กระทั่งในรายที่ไม่เคยสูบบุหรี่เลย </h4>
                            <h4 style='text-align: center; color: black; font-size: 150%'> โดยจะพบเซลลมะเร็งชนิดนี้ได้ที่ต่อมสร้างน้ำเมือกของปอดหรือถุงลม </h4>
                        </div>
                            """, unsafe_allow_html=True)   
            elif out_load[0] == 'normal':
                st.markdown(
                        """
                    <div style='border: 2px solid green; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: green; font-size: 180%'> Normal </h3>
                    </div>
                        """, unsafe_allow_html=True) 
                st.markdown(
                            """
                        <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h3 style='text-align: center; color: black; font-size: 180%'> ปกติ </h3>
                        </div>
                            """, unsafe_allow_html=True)     
            elif out_load[0] == 'large.cell':
                st.markdown(
                        """
                    <div style='border: 2px solid orange; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: orange; font-size: 180%'> Large.Cell </h3>
                    </div>
                        """, unsafe_allow_html=True)   
                st.markdown(
                            """
                        <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h4 style='text-align: center; color: black; font-size: 150%'> มักจะพบเซลล์ชนิดนี้ที่ผิวนอกของเนื้อปอด บริเวณริมขอบ และแพร่กระจายได้เร็วมาก </h4>
                            <h4 style='text-align: center; color: black; font-size: 150%'> พบประมาณ 5-15% ของคนที่เป็นมะเร็งปอดทั้งหมด </h4>
                        </div>
                            """, unsafe_allow_html=True)     
            elif out_load[0] == 'squamous.cell':
                st.markdown(
                        """
                    <div style='border: 2px solid #f16565; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: #f16565; font-size: 180%'> Squamous.Cell </h3>
                    </div>
                        """, unsafe_allow_html=True)  
                st.markdown(
                            """
                        <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h4 style='text-align: center; color: black; font-size: 150%'> จะพบได้ที่เยื่อบุผิวหลอดลม ประมาณ 30-35% ของคนที่เป็นมะเร็งปอดทั้งหมด  โดยเฉพาะในผู้ชายและผู้สูงอายุทั้งชายและหญิง </h4>
                            <h4 style='text-align: center; color: black; font-size: 150%'> เกี่ยวข้องอย่างมากกับการสูบบุหรี่ ทำให้ไอมาก อาจไอเป็นเลือดหรือปอดบวม </h4>
                            <h4 style='text-align: center; color: black; font-size: 150%'> เพราะก้อนเนื้อมะเร็งไปอุดท่อของหลอดลมจนทำให้ไม่สามารถไอเอาเสมหะออกมาได้ </h4>
                        </div>
                            """, unsafe_allow_html=True)  
            st.image(dff_image, use_column_width=True)
            st.image(gradcam_image, use_column_width=True)
            st.balloons()


elif tabs == 'Analytics' and count_system() == 1:
    data_base = []  
    with open('system.txt', 'r') as f:
        # Read the dictionary from the file
        dictionary_str = f.readline().strip()
        data_dict = eval(dictionary_str)  # Convert the string to a dictionary using eval()
        data_base.append(data_dict)  # Append the dictionary to the data_base list
    f.close() 
    option = {
    "tooltip": {
    "trigger": 'axis',
    "axisPointer": {
    # Use axis to trigger tooltip
    "type": 'shadow' # 'shadow' as default; can also be 'line' or 'shadow'
        }
    },
    "legend": {},
    "grid": {
    "left": '3%',
    "right": '4%',
    "bottom": '3%',
    "containLabel": True
    },
        "xAxis": {
        "type": 'value'
    },
        "yAxis": {
        "type": 'category',
        "data": ['Results']
        },
        "series": [
    {
        "name": 'Adenocarcinoma',
        "type": 'bar',
        "stack": 'total',
        "label": {
        "show": True
    },
        "emphasis": {
            "focus": 'series'
        },
            "data": ['{:.4f}'.format(data_base[0]['adenocarcinoma'])]
        },
            {
            "name": 'Normal',
            "type": 'bar',
            "stack": 'total',
            "label": {
            "show": True
        },
        "emphasis": {
            "focus": 'series'
        },
        "data": ['{:.4f}'.format(data_base[0]['normal'])]
        },
        {
        "name": 'Large.Cell',
        "type": 'bar',
        "stack": 'total',
        "label": {
            "show": True
        },
        "emphasis": {
            "focus": 'series'
        },
        "data": ['{:.4f}'.format(data_base[0]['large.cell'])]
        },
        {
        "name": 'Squamous.Cell',
        "type": 'bar',
        "stack": 'total',
        "label": {
            "show": True
        },
        "emphasis": {
        "focus": 'series'
        },
        "data": ['{:.4f}'.format(data_base[0]['squamous.cell'])]
        },
    ]
}
    st_echarts(options=option)  
    out_load = []
    with open('className.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out_load.append(line)
    f.close() 
    if out_load[0] == 'adenocarcinoma':
        st.markdown(
                    """
                <div style='border: 2px solid #5370c6; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: #5370c6; font-size: 180%'> Adenocarcinoma </h3>
                </div>
                    """, unsafe_allow_html=True)   
    elif out_load[0] == 'normal':
        st.markdown(
                    """
                <div style='border: 2px solid green; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: green; font-size: 180%'> Normal </h3>
                </div>
                    """, unsafe_allow_html=True)     
    elif out_load[0] == 'large.cell':
        st.markdown(
                    """
                <div style='border: 2px solid orange; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: orange; font-size: 180%'> Large.Cell </h3>
                </div>
                    """, unsafe_allow_html=True)     
    elif out_load[0] == 'squamous.cell':
        st.markdown(
                    """
                <div style='border: 2px solid #f16565; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: #f16565; font-size: 180%'> Squamous.Cell </h3>
                </div>
                    """, unsafe_allow_html=True)  
    dff_image_path = r'.\save_images\dff_image.png'
    def display_images(dff_image_path):
        gradcam_image = Image.open(dff_image_path)
        st.image(gradcam_image, use_column_width=True)
    display_images(dff_image_path)
    if st.button("Refresh",use_container_width=True):
            os.remove('.\save_images\dff_image.png')
            os.remove('.\save_images\gradcam_image.png')
            with open('system.txt', 'w') as file:
                file.write('')
            file.close() 
            with open('className.txt', 'w') as file:
                file.write('')
            file.close() 
            st.experimental_rerun()
            # open the images

elif tabs == 'Analytics' and count_system() == 0:
        st.markdown(
                """
            <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                <h3 style='text-align: center; color: red; font-size: 180%'> 🖼️ Image Analytics Not Detected ❌ </h3>
            </div>
                """, unsafe_allow_html=True)
elif tabs == 'Upload' and count_system() == 1:
    dff_image_path = r'.\save_images\dff_image.png'
    gradcam_image_path = r'.\save_images\gradcam_image.png'
    def display_images(dff_image_path, gradcam_image_path):
        out_load = []
        with open('className.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out_load.append(line)
        f.close() 
        if out_load[0] == 'adenocarcinoma':
            st.markdown(
                    """
                <div style='border: 2px solid #5370c6; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: #5370c6; font-size: 180%'> Adenocarcinoma </h3>
                </div>
                    """, unsafe_allow_html=True)   
            st.markdown(
                        """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h4 style='text-align: center; color: black; font-size: 150%'> พบประมาณ 40% ของคนที่เป็นมะเร็งปอดทั้งหมดที่เป็นผู้ชายและผู้หญิง แม้กระทั่งในรายที่ไม่เคยสูบบุหรี่เลย </h4>
                    <h4 style='text-align: center; color: black; font-size: 150%'> โดยจะพบเซลลมะเร็งชนิดนี้ได้ที่ต่อมสร้างน้ำเมือกของปอดหรือถุงลม </h4>
                </div>
                        """, unsafe_allow_html=True)   
        elif out_load[0] == 'normal':
            st.markdown(
                    """
                <div style='border: 2px solid green; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: green; font-size: 180%'> Normal </h3>
                </div>
                    """, unsafe_allow_html=True) 
            st.markdown(
                        """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: black; font-size: 180%'> ปกติ </h3>
                </div>
                        """, unsafe_allow_html=True)     
        elif out_load[0] == 'large.cell':
            st.markdown(
                    """
                <div style='border: 2px solid orange; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: orange; font-size: 180%'> Large.Cell </h3>
                </div>
                    """, unsafe_allow_html=True)   
            st.markdown(
                        """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h4 style='text-align: center; color: black; font-size: 150%'> มักจะพบเซลล์ชนิดนี้ที่ผิวนอกของเนื้อปอด บริเวณริมขอบ และแพร่กระจายได้เร็วมาก </h4>
                    <h4 style='text-align: center; color: black; font-size: 150%'> พบประมาณ 5-15% ของคนที่เป็นมะเร็งปอดทั้งหมด </h4>
                </div>
                        """, unsafe_allow_html=True)     
        elif out_load[0] == 'squamous.cell':
            st.markdown(
                    """
                <div style='border: 2px solid #f16565; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: #f16565; font-size: 180%'> Squamous.Cell </h3>
                </div>
                    """, unsafe_allow_html=True)  
            st.markdown(
                            """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h4 style='text-align: center; color: black; font-size: 150%'> จะพบได้ที่เยื่อบุผิวหลอดลม ประมาณ 30-35% ของคนที่เป็นมะเร็งปอดทั้งหมด  โดยเฉพาะในผู้ชายและผู้สูงอายุทั้งชายและหญิง </h4>
                    <h4 style='text-align: center; color: black; font-size: 150%'> เกี่ยวข้องอย่างมากกับการสูบบุหรี่ ทำให้ไอมาก อาจไอเป็นเลือดหรือปอดบวม </h4>
                    <h4 style='text-align: center; color: black; font-size: 150%'> เพราะก้อนเนื้อมะเร็งไปอุดท่อของหลอดลมจนทำให้ไม่สามารถไอเอาเสมหะออกมาได้ </h4>
                </div>
                            """, unsafe_allow_html=True)  
            st.image(dff_image, use_column_width=True)
            st.image(gradcam_image, use_column_width=True)
            st.balloons()
        if st.button("Refresh",use_container_width=True):
            os.remove('.\save_images\dff_image.png')
            os.remove('.\save_images\gradcam_image.png')
            with open('system.txt', 'w') as file:
                file.write('')
            file.close() 
            with open('className.txt', 'w') as file:
                file.write('')
            file.close() 
            st.experimental_rerun()
        # open the images
        dff_image = Image.open(dff_image_path)
        gradcam_image = Image.open(gradcam_image_path)
        # display the images on Streamlit
        st.markdown(
                """
            <div style='border: 2px solid red; border-radius: 15px; padding: 5px; background-color: white;'>
                <h1 style='text-align: center; color: black;'> DFF Image </h1>
            </div>
                """, unsafe_allow_html=True)
        st.image(dff_image, use_column_width=True)
        st.markdown(
                """
            <div style='border: 2px solid red; border-radius: 15px; padding: 5px; background-color: white;'>
                <h1 style='text-align: center; color: black;'> Gradcam Image </h1>
            </div>
                """, unsafe_allow_html=True)
        st.image(gradcam_image, use_column_width=True)
    display_images(dff_image_path, gradcam_image_path)

elif tabs == 'More Information':
    out_load = []
    with open('className.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out_load.append(line)
    f.close() 
    if out_load == []:
        st.markdown(
                    """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: red; font-size: 180%'> Non-find </h3>
                </div>
                    """, unsafe_allow_html=True)   
    elif out_load[0] == 'adenocarcinoma':
        gradcam_image_path = r'.\save_images\gradcam_image.png'
        def display_images(gradcam_image_path):
            if st.button("Refresh",use_container_width=True):
                os.remove('.\save_images\dff_image.png')
                os.remove('.\save_images\gradcam_image.png')
                with open('system.txt', 'w') as file:
                    file.write('')
                file.close() 
                with open('className.txt', 'w') as file:
                    file.write('')
                file.close() 
                st.experimental_rerun()
            # open the images
            gradcam_image = Image.open(gradcam_image_path)
            st.image(gradcam_image, use_column_width=True)
        display_images(gradcam_image_path)
        st.markdown(
                    """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: red; font-size: 180%'> Adenocarcinoma </h3>
                </div>
                    """, unsafe_allow_html=True)          
        st.markdown(
                    """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: red; font-size: 180%'> Lung adenocarcinoma is a type of non-small cell lung cancer that begins in the glandular cells of the lung. It is the most common type of lung cancer and is often found in people who have never smoked. Here are some key characteristics of lung adenocarcinoma </h3>
                </div>
                    """, unsafe_allow_html=True)         
        st.markdown(
                    """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: red; font-size: 180%'> Adenocarcinoma </h3>
                </div>
                    """, unsafe_allow_html=True)    
    elif out_load[0] == 'large.cell':
        st.markdown(
                    """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: red; font-size: 180%'> Large.Cell </h3>
                </div>
                    """, unsafe_allow_html=True)     
    elif out_load[0] == 'normal':
        st.markdown(
                    """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: red; font-size: 180%'> Normal </h3>
                </div>
                    """, unsafe_allow_html=True)   
    elif out_load[0] == 'squamous.cell':
        st.markdown(
                    """
                <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: red; font-size: 180%'> Squamous.Cell </h3>
                </div>
                    """, unsafe_allow_html=True)    
               
            
