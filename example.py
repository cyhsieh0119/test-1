from PIL import Image
import streamlit as st
import numpy as np
import torch
#import scipy.stats as st
import plotly.express as px
from matplotlib import pyplot as plt

st.set_page_config(page_title="Image converter" ,page_icon="random" ,layout="wide")

st.markdown("# Image converter page 🎈")
st.sidebar.markdown("# Main page 🎈 of Side Bar")


def main():
	    st.title("File Upload Tutorial")
	    menu = ["Image","Dataset","DocumentFiles","About"]
	    choice = st.sidebar.selectbox("Menu",menu)

	    if choice == "Image":
		st.subheader("Image")

	    elif choice == "Dataset":
		st.subheader("Dataset")

	    elif choice == "DocumentFiles":
		st.subheader("DocumentFiles")

def load_image(image_file):
	    img = Image.open(image_file)
	    return img

if choice == "Image":
	    st.subheader("Image")
	    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

	    if image_file is not None:
		# To See details
		file_details = {"filename":image_file.name, "filetype":image_file.type,
				"filesize":image_file.size}
		st.write(file_details)
		# To View Uploaded Image
		st.image(load_image(image_file),width=250)

if __name__ == '__main__':
	main()
