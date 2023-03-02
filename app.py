import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

import streamlit as st

labels = ["adenocarcinoma","large.cell.carcinoma","normal","squamous.cell.carcinoma"]
model_name_or_path = 'alicelouis/ViTLungMiNi'


@st.cache(allow_output_mutation=True,show_spinner=False,ttl=1800,max_entries=2,persist=True)
def FeatureExtractor(model_name_or_path):
    feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
    return feature_extractor


@st.cache(allow_output_mutation=True,show_spinner=False,ttl=1800,max_entries=2,persist=True)
def LoadModel(model_name_or_path):
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True)
    return model
    
# end def
with st.sidebar:

    st.header("🖥️เกี่ยวกับโปรเจคนี้")
    st.write("ยังไม่เสร็จ",unsafe_allow_html=True)
    
    st.header("🌐แหล่งอ้างอิง")
    st.write("ยังไม่เสร็จ")

st.header('Lung Cancer classification with Vision Transformer: จำแนกมะเร็งปอด')


with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
hide_table_index = """
            <style>         
            thead {display:none}  
            tbody th {display:none}
            .blank {display:none}
            </style>
            """ 
st.markdown(hide_table_index, unsafe_allow_html=True)


    
uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ")

if uploaded_file is not None:
    model_name_or_path = 'alicelouis/ViTLungMiNi'
    feature_extractor = FeatureExtractor(model_name_or_path)
    model = LoadModel(model_name_or_path)
    img = Image.open(uploaded_file)
    img_out = img
    img_out = np.array(img_out)
    # โหลดโมเดลที่เซฟ
    image = img.resize((224,224))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 4 classes
    predicted_class_idx = logits.argmax(-1).item()
    className = labels[predicted_class_idx]

    st.success("Predicted class is : " + className , icon="✅")
    st.image(img_out)
