from PIL import Image
import cv2
import streamlit as st
import numpy as np
import torch
import scipy.stats as sst
from matplotlib import pyplot as plt

st.set_page_config(page_title="Image converter" ,page_icon="random" ,layout="wide")

st.markdown("# Image converter page 🎈")
st.sidebar.markdown("# Main page 🎈 of Side Bar")

def initD():
	model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid   (medium accuracy, medium inference speed)

	midas = torch.hub.load("intel-isl/MiDaS", model_type)

	#
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	midas.to(device)
	midas.eval()
	#
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

	if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
		transform = midas_transforms.dpt_transform
	else:
		transform = midas_transforms.small_transform
	
def depthRaw(img0):
	#
	model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid   (medium accuracy, medium inference speed)
	midas = torch.hub.load("intel-isl/MiDaS", model_type)
	#
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	midas.to(device)
	midas.eval()
	#
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
	if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
		transform = midas_transforms.dpt_transform
	else:
		transform = midas_transforms.small_transform
	#
	input_batch = transform(img0).to(device)
	#
	with torch.no_grad():
		prediction = midas(input_batch)
	#
		prediction = torch.nn.functional.interpolate(
				prediction.unsqueeze(1),
				size=img0.shape[:2],
				mode="bicubic",
				align_corners=False,
				).squeeze()
	output = prediction.cpu().numpy()
	return output

def load_image(image_file):
	#img = cv2.imread(image_file)
	img = Image.open(image_file)
	return img

def main():
	st.title("File Upload Tutorial")
	menu = ["Image","Dataset","DocumentFiles","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	tab1, tab2, tab3 = st.tabs(["img1", "Dog", "Owl"])

#    with tab1:
#        st.header("A cat")
#        st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

	if choice == "Image":
		st.subheader("Image")
		image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
		if image_file is not None:
			# To See details
			img0=load_image(image_file)
			img = img0.save(image_file.name)
			frame = cv2.imread(image_file.name)
			img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			output=depthRaw(img1)
			file_details = {"filename":image_file.name, "filetype":image_file.type,
					"filesize":image_file.size}
			st.sidebar.write(file_details)
			# To View Uploaded Image
			with tab1:
				st.header(image_file.name)
				st.image(img0,width=1000)
				

	elif choice == "Dataset":
		st.subheader("Dataset")

	elif choice == "DocumentFiles":
		st.subheader("DocumentFiles")




if __name__ == '__main__':
	#initD()
	main()
