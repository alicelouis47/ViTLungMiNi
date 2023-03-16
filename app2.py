import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_echarts import st_echarts
from streamlit_elements import elements, mui, html
# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    print(selected)
    selected
    

list_to_be_sorted= [
        { "value": 335, "name": 'Direct' },
        { "value": 310, "name": 'Email' },
        { "value": 274, "name": 'Union Ads' },
        { "value": 235, "name": 'Video Ads' },
        { "value": 400, "name": 'Search Engine' }]
newlist = sorted(list_to_be_sorted, key=lambda d: d['value']) 
selected2 = option_menu(None, ["Home", "Upload"], 
    icons=['house', 'cloud-upload'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
if(selected == "Home"):
        st.header("🖥️เกี่ยวกับโปรเจคนี้")
        st.write("ยังไม่เสร็จ",unsafe_allow_html=True)
        options = {
  "backgroundColor": '#2c343c',
  "title": {"text": 'Customized Pie',"left": 'center',"top": 20,"textStyle": {"color": '#ccc'}},
  "tooltip": {"trigger": 'item'},
  "visualMap": {
    "show": False,
    "min": 80,
    "max": 600,
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
        "length2": 20
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
elif(selected =="Settings"):
        st.header("🌐แหล่งอ้างอิง")
        st.write("ยังไม่เสร็จ")
        st.write('Hello, *World!* :sunglasses:')
        with elements("multiple_children"):
            with mui.Paper(elevation=3, variant="outlined", square=True):
                mui.TextField(
            label="My text input",
            defaultValue="Type here",
            variant="outlined",
        )
            mui.Card (variant="outlined")
            mui.Button(
        mui.icon.EmojiPeople,
        mui.icon.DoubleArrow,
        "Button with multiple children"
    )

 
            with mui.Button:
                mui.icon.EmojiPeople()
                mui.icon.DoubleArrow()
                mui.Typography("Button with multiple children")
            with mui.Paper:
                with mui.Typography:
                    html.p("Hello world")
                    html.p("Goodbye world")
            mui.Box(
        "Some text in a styled box",
        sx={
            "bgcolor": "background.paper",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
            "minWidth": 300,
        }
    )