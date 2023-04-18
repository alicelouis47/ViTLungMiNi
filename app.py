import numpy as np
from transformers import BeitImageProcessor, BeitForImageClassification
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
import pydicom

labels = ["adenocarcinoma","large.cell","normal","squamous.cell"]
model_name_or_path = 'alicelouis/BeiT_NSCLC_lr2e-5'
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

@st.cache_resource(show_spinner=False,ttl=1800,max_entries=2)
def FeatureExtractor(model_name_or_path):
    feature_extractor = BeitImageProcessor.from_pretrained(model_name_or_path)
    return feature_extractor


@st.cache_resource(show_spinner=False,ttl=1800,max_entries=2)
def LoadModel(model_name_or_path):
    model = BeitForImageClassification.from_pretrained(
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
    count_system = []
    with open('count_class.txt', 'r') as f:
        for line in f:
            if line.strip() == '0':
                continue
            else:
                count_system.append(line.strip())
    f.close()
    if len(count_system) != 0:
        return int(len(count_system))
    elif len(count_system) == 0:
        return int(0)


def count_class(count_classes):
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(len(count_classes)):
        if count_classes[i] == "Adeno":
            a += 1
        elif count_classes[i] == "Normal":
            b += 1
        elif count_classes[i] == "Large":
            c += 1
        elif count_classes[i] == "Squamous":
            d += 1 
    count_classes = []
    count_classes.append(str(a))
    count_classes.append(str(b))
    count_classes.append(str(c))
    count_classes.append(str(d))
    with open("count_class.txt", "w") as f:
        for count in count_classes:
            f.write(count + "\n")

# Define CSS styling for centering
centered_style = """
        display: flex;
        justify-content: center;
"""

st.markdown(
    """
<div style='border: 2px solid green; border-radius: 5px; padding: 10px; background-color: white;'>
    <h1 style='text-align: center; color: green;'>
    🏥 Lung Cancer Classification with Vision Transformer : จำแนกมะเร็งปอด 🫁
    </h1>
</div>
    """, unsafe_allow_html=True)

with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
# end def

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home','Upload', 'Analytics', 'More Information', 'Reset'], 
    iconName=['home','upload', 'analytics', 'informations', 'refresh'], 
    styles={'navtab': {'background-color': '#111', 'color': '#818181', 'font-size': '18px', 
                    'transition': '.3s', 'white-space': 'nowrap', 'text-transform': 'uppercase'}, 
                    'tabOptionsStyle': 
                    {':hover :hover': {'color': 'red', 'cursor': 'pointer'}}, 'iconStyle': 
                    {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'}, 'tabStyle': 
                    {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'}}, 
                    key="1",default_choice=0)
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
elif tabs == 'Upload': #and count_system () != 1:
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ", type=["jpg", "jpeg", "png", "dcm"], accept_multiple_files=True)
    name_of_files = []
    name_of_files_new = []
    for n in uploaded_file:
        file_name = n.name
        name_of_files.append(file_name)
    with open("save_name.txt", "w") as f:
        for name in name_of_files:
            f.write(name + "\n")
    for j in range(len(name_of_files)):
        if name_of_files[j].endswith('.dcm'):
            name_of_files_new.append(name_of_files[j][:-4] + '.png')
        else:
            name_of_files_new.append(name_of_files[j])  
    for i in range(len(uploaded_file)):   
        if name_of_files[i].endswith('.dcm'):
            ds = pydicom.dcmread(uploaded_file[i])
            new_image = ds.pixel_array.astype(float)
            scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
            scaled_image = np.uint8(scaled_image)
            gray_scale = Image.fromarray(scaled_image)
            final_image = gray_scale.convert('RGB')
            final_image.resize((200,200))
            final_image.save(r'.\dcm_png\{}.png'.format(name_of_files[i]))                  
        feature_extractor = FeatureExtractor(model_name_or_path)
        model = LoadModel(model_name_or_path)
        if name_of_files[i].endswith('.dcm'):
            img = Image.open(r'.\dcm_png\{}.png'.format(name_of_files[i]))
        else:
            img = Image.open(uploaded_file[i])
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
        target_layer_dff = model.beit.layernorm
        target_layer_gradcam = model.beit.encoder.layer[-2].output
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
        # dff_image.save(r".\save_images\dff_image.png")
        # gradcam_image.save(r".\save_images\gradcam_image.png")
        topK = print_top_categories(model, tensor_resized)
        df = pd.DataFrame.from_dict(topK, orient='index')
        list_to_be_sorted= []
        for x, y in topK.items():
            dic = dict()
            dic["value"] = y
            dic["name"] = x
            list_to_be_sorted.append(dic)
            data_base.append(y)
        if list_to_be_sorted[0]['name'] == "adenocarcinoma":
            dff_image.save(r"./Adenocarcinoma/{}".format(name_of_files_new[i]))
            image_path = name_of_files_new[i]
            with Image.open(r"./Adenocarcinoma/{}".format(image_path)) as image:
                width, height = image.size
                new_width = 2 * width // 3
                cropped_image = image.crop((0, 0, new_width, height))
                cropped_image.save(r"./Adenocarcinoma/{}".format(image_path))
        elif list_to_be_sorted[0]['name'] == "large.cell":
            dff_image.save(r"./Large cell carcinoma/{}".format(name_of_files_new[i]))
            image_path = name_of_files_new[i]
            with Image.open(r"./Large cell carcinoma/{}".format(image_path)) as image:
                width, height = image.size
                new_width = 2 * width // 3
                cropped_image = image.crop((0, 0, new_width, height))
                cropped_image.save(r"./Large cell carcinoma/{}".format(image_path))
            #dff_image.save(r".\Large cell carcinoma\{}".format(name_of_files_new[i]))
        elif list_to_be_sorted[0]['name'] == "normal":
            dff_image.save(r"./Normal/{}".format(name_of_files_new[i]))
            image_path = name_of_files_new[i]
            with Image.open(r"./Normal/{}".format(image_path)) as image:
                width, height = image.size
                new_width = 2 * width // 3
                cropped_image = image.crop((0, 0, new_width, height))
                cropped_image.save(r"./Normal/{}".format(image_path))
            #dff_image.save(r".\Normal\{}".format(name_of_files_new[i]))
        elif list_to_be_sorted[0]['name'] == "squamous.cell":
            dff_image.save(r"./Squamous cell carcinoma/{}".format(name_of_files_new[i]))
            image_path = name_of_files_new[i]
            with Image.open(r"./Squamous cell carcinoma/{}".format(image_path)) as image:
                width, height = image.size
                new_width = 2 * width // 3
                cropped_image = image.crop((0, 0, new_width, height))
                cropped_image.save(r"./Squamous cell carcinoma/{}".format(image_path))
            #dff_image.save(r".\Squamous cell carcinoma\{}".format(name_of_files_new[i]))
        # st.image(dff_image, use_column_width=True)
        # st.image(gradcam_image, use_column_width=True)
        st.balloons()

    # Create a container for the two columns
    container = st.container()
    # Create two columns within the container
    col1, col2 = container.columns(2)
    col3, col4 = container.columns(2)
    col5, col6 = container.columns(2)
    # Add the first subheader to the first column
    count_classes = [] #Adenocarcinoma, Normal, Large cell carcinoma, Squamous cell carcinoma
    with col1:
        st.markdown("<h2 style='text-align: center; border: 2px solid #5370c6; border-radius: 5px; padding: 15px; background-color: white; color: black;' > Adenocarcinoma </h2>".format(centered_style), unsafe_allow_html=True)
        # Add the second subheader to the second column
        folder_path = r".\Adenocarcinoma"
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
        # Display the images in a loop
        for i in range(0, len(image_files), 2):
            col7, col8 = st.columns([1, 1])
            with col7:
                if i < len(image_files):
                    image1 = Image.open(os.path.join(folder_path, image_files[i]))
                    st.image(image1, use_column_width=True)
                    st.write(f"<p style='text-align: center; color: black; border: 2px solid white; border-radius: 10px; padding: 10px; background-color: #5370c6; font-size: 32px;'>{image_files[i]}</p>", unsafe_allow_html=True)
                    count_classes.append("Adeno")
            with col8:
                if i+1 < len(image_files):
                    image2 = Image.open(os.path.join(folder_path, image_files[i+1]))
                    st.image(image2, use_column_width=True)
                    st.write(f"<p style='text-align: center; color: black; border: 2px solid white; border-radius: 10px; padding: 10px; background-color: #5370c6; font-size: 32px;'>{image_files[i+1]}</p>", unsafe_allow_html=True) 
                    count_classes.append("Adeno")
    with col2:
        st.markdown("<h2 style='text-align: center; border: 2px solid green; border-radius: 5px; padding: 15px; background-color: white; color: black;' > Normal </h2>".format(centered_style), unsafe_allow_html=True)
        folder_path = r".\Normal"
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
        # Display the images in a loop
        for i in range(0, len(image_files), 2):
            col9, col10 = st.columns([1, 1])
            with col9:
                if i < len(image_files):
                    image1 = Image.open(os.path.join(folder_path, image_files[i]))
                    st.image(image1, use_column_width=True)
                    st.write(f"<p style='text-align: center; color: black; border: 2px solid white; border-radius: 10px; padding: 10px; background-color: green; font-size: 32px;'>{image_files[i]}</p>", unsafe_allow_html=True)
                    count_classes.append("Normal")
            with col10:
                if i+1 < len(image_files):
                    image2 = Image.open(os.path.join(folder_path, image_files[i+1]))
                    st.image(image2, use_column_width=True) 
                    st.write(f"<p style='text-align: center; color: black; border: 2px solid white; border-radius: 10px; padding: 10px; background-color: green; font-size: 32px;'>{image_files[i+1]}</p>", unsafe_allow_html=True)   
                    count_classes.append("Normal")
    with col3:
        st.markdown("")
    with col4:
        st.markdown("")

    with col5:
        st.markdown("<h2 style='text-align: center; border: 2px solid orange; border-radius: 5px; padding: 15px; background-color: white; color: black;' > Large cell carcinoma </h2>".format(centered_style), unsafe_allow_html=True)
        folder_path = r".\Large cell carcinoma"
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
        # Display the images in a loop
        for i in range(0, len(image_files), 2):
            col11, col12 = st.columns([1, 1])
            with col11:
                if i < len(image_files):
                    image1 = Image.open(os.path.join(folder_path, image_files[i]))
                    st.image(image1, use_column_width=True)
                    st.write(f"<p style='text-align: center; color: black; border: 2px solid white; border-radius: 10px; padding: 10px; background-color: orange; font-size: 32px;'>{image_files[i]}</p>", unsafe_allow_html=True)   
                    count_classes.append("Large")
            with col12:
                if i+1 < len(image_files):
                    image2 = Image.open(os.path.join(folder_path, image_files[i+1]))
                    st.image(image2, use_column_width=True)
                    st.write(f"<p style='text-align: center; color: black; border: 2px solid white; border-radius: 10px; padding: 10px; background-color: orange; font-size: 32px;'>{image_files[i+1]}</p>", unsafe_allow_html=True)   
                    count_classes.append("Large")
    with col6:
        st.markdown("<h2 style='text-align: center; border: 2px solid #f16565; border-radius: 5px; padding: 15px; background-color: white; color: black;' >  Squamous cell carcinoma </h2>".format(centered_style), unsafe_allow_html=True)
        folder_path = r".\Squamous cell carcinoma"
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
        # Display the images in a loop
        for i in range(0, len(image_files), 2):
            col13, col14 = st.columns([1, 1])
            with col13:
                if i < len(image_files):
                    image1 = Image.open(os.path.join(folder_path, image_files[i]))
                    st.image(image1, use_column_width=True)
                    st.write(f"<p style='text-align: center; color: black; border: 2px solid white; border-radius: 10px; padding: 10px; background-color: #f16565; font-size: 32px;'>{image_files[i]}</p>", unsafe_allow_html=True)   
                    count_classes.append("Squamous")
            with col14:
                if i+1 < len(image_files):
                    image2 = Image.open(os.path.join(folder_path, image_files[i+1]))
                    st.image(image2, use_column_width=True)
                    st.write(f"<p style='text-align: center; color: black; border: 2px solid white; border-radius: 10px; padding: 10px; background-color: #f16565; font-size: 32px;'>{image_files[i+1]}</p>", unsafe_allow_html=True)   
                    count_classes.append("Squamous")
    count_class(count_classes)

elif tabs == 'Analytics' and count_system() > 0:
    data_base = []
    data_base_max = []
    #max_value = max(data_base)
    #max_index = data_base.index(max_value)
    with open('count_class.txt', 'r') as f:
        for line in f:
            data_base.append(line.strip()) 
            data_base_max.append(int(line.strip()))
    max_value = max(data_base_max)  # Find the maximum value in the list
    max_index = data_base_max.index(max_value)
    max_indices = [i for i, value in enumerate(data_base_max) if value == max_value]
    if len(max_indices) > 1:
        max_index = 4
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
            "data": [data_base[0]]
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
        "data": [data_base[1]]
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
        "data": [data_base[2]]
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
        "data": [data_base[3]]
        },
    ]
}
    st_echarts(options=option)
    if max_index == 0:
        st.markdown("<h2 style='text-align: center; border: 2px solid #5370c6; border-radius: 5px; padding: 15px; background-color: white; color: black;' > Adenocarcinoma </h2>".format(centered_style), unsafe_allow_html=True)
    elif max_index == 1:
        st.markdown("<h2 style='text-align: center; border: 2px solid green; border-radius: 5px; padding: 15px; background-color: white; color: black;' > Normal </h2>".format(centered_style), unsafe_allow_html=True)
    elif max_index == 2:
        st.markdown("<h2 style='text-align: center; border: 2px solid orange; border-radius: 5px; padding: 15px; background-color: white; color: black;' > Large cell carcinoma </h2>".format(centered_style), unsafe_allow_html=True)
    elif max_index == 3:   
        st.markdown("<h2 style='text-align: center; border: 2px solid #f16565; border-radius: 5px; padding: 15px; background-color: white; color: black;' >  Squamous cell carcinoma </h2>".format(centered_style), unsafe_allow_html=True)

elif tabs == 'Analytics' and count_system() == 0:
        st.markdown(
                """
            <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                <h3 style='text-align: center; color: red; font-size: 180%'> 🖼️ Image Analytics Not Detected ❌ </h3>
            </div>
                """, unsafe_allow_html=True)
        
elif tabs == 'More Information':
        st.markdown(
                    """
                <div style='border: 2px dashed blue; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: black; font-size: 180%'> 💻 Organizers 🖱️ </h3>
                </div>
                    """, unsafe_allow_html=True)          
        st.markdown(
            """
                <div style="display:flex; justify-content:center; align-items:center;">
                <img src="https://drive.google.com/uc?export=view&id=1xupbYYXQZzjwMQiVGwT636oCXMga2ETF" style="width:300px; height:200px; margin: 10px;">
                <img src="https://drive.google.com/uc?export=view&id=1evDy9sDtJ1T_WVR1bUnfyZkeSMjT9pfr" style="width:300px; height:200px; margin: 10px;">
                <img src="https://drive.google.com/uc?export=view&id=1Sebh31aX8vdNe8P7oyBL714J_0qA5WYt" style="width:300px; height:200px; margin: 10px;">
                </div>
            """, unsafe_allow_html=True)
        st.markdown(
            """
                <div style="display:flex; justify-content:center; align-items:center;">
                <h3 style="width:300px; height:200px; margin: 10px; font-size: 50% text-align: center;' "> 👑 Santipab Tongchan\nCall : 090-2471512 \n "stdm4522@pccbr.ac.th" </h3>
                <h3 style="width:300px; height:200px; margin: 10px; font-size: 50% text-align: center;' "> Phakkhaphon Artburai\nCall : 091-0197314 \n "stdm4321@pccbr.ac.th" </h3>
                <h3 style="width:300px; height:200px; margin: 10px; font-size: 50% text-align: center;' "> Natthawee Naewkumpol\nCall : 061-9487722 \n "stdm4605@pccbr.ac.th" </h3>
                </div>
            """, unsafe_allow_html=True)
        st.markdown(
                    """
                <div style='border: 2px solid orange; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: blue; font-size: 200%'> Princess Chulabhorn Science High School Buriram </h3>
                </div>
                    """, unsafe_allow_html=True)    

elif tabs == 'Reset':
    def clear_folder(folder_name):
        # Check if the folder exists
        if not os.path.exists(folder_name):
            print(f"{folder_name} does not exist.")
            return
        # Get a list of all files in the folder and its subdirectories
        files = []
        for dirpath, dirnames, filenames in os.walk(folder_name):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))

        # Delete all files in the list
        for file in files:
            os.remove(file)
    clear_folder('Adenocarcinoma')
    clear_folder('Large cell carcinoma')
    clear_folder('Normal')
    clear_folder('Squamous cell carcinoma')
    clear_folder('dcm_png')
    #clear data in count_class
    with open('count_class.txt', 'w') as file:
        file.write('')
    st.markdown(
            """
        <div style='border: 2px solid #00FFFF; border-radius: 5px; padding: 5px; background-color: white;'>
            <h3 style='text-align: center; color: blue; font-size: 180%'> 🔃 The information has been cleared. ✅ </h3>
        </div>
            """, unsafe_allow_html=True)
