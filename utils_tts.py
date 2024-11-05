import datetime
import requests
import streamlit as st
import pytz


# --------- huggingface api inference for text to speech ----------#

def txt2speech(text):
    print("Initializing text-to-speech conversion...")
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {
        "Authorization": f'Bearer {st.secrets["huggingfacehub_api_token"]}'}
    payloads = {'inputs': text}

    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.mp3', 'wb') as file:
        file.write(response.content)


# --------- bucket time for chatbot greetings  ----------#

def get_time_bucket():

    gmt = pytz.timezone('GMT')
    now_gmt = datetime.datetime.now(gmt)
    hour = now_gmt.hour  # now = datetime.datetime.now()
    sg_time = 8  # Singapore is UTC+8

    if hour + sg_time < 12:
        return "Morning greetings!"
    elif hour + sg_time < 17:
        return "Good afternoon!"
    else:
        return "Good evening!"


# custom CSS for buttons
custom_css = """
<style>
    .stButton > button {
        color: #383736; 
        border: none; /* No border */
        padding: 5px 22px; /* Reduced top and bottom padding */
        text-align: center; /* Centered text */
        text-decoration: none; /* No underline */
        display: inline-block; /* Inline-block */
        font-size: 12px !important;
        margin: 4px 2px; /* Some margin */
        cursor: pointer; /* Pointer cursor on hover */
        border-radius: 30px; /* Rounded corners */
        transition: background-color 0.3s; /* Smooth background transition */
    }
    .stButton > button:hover {
        color: #383736; 
        background-color: #c4c2c0; /* Darker green on hover */
    }
</style>
"""

# ------ set up question button -----#
example_prompts = [
    "News from ChannelNewsAsia",
    "News from MustShareNews",
    "Weather forecast next few days",
    "Weather forecast today",
    "Translate in Chinese",
    "An ultra-realistic image of an astronaut on a crowded bus",
    "Billie Eilish - BIRDS OF A FEATHER (official music video)",

]


# --------- lottie spinner  ----------#


# def load_lottieurl(url: str):
#    r = requests.get(url)
#    if r.status_code != 200:
#        return None
#    return r.json()
#
#
# lottie_bookflip_download = load_lottieurl(
#    "https://lottie.host/71eb8ff6-9973-4ab0-b2c5-2de92fa51183/jJGCTnVWTb.json")
