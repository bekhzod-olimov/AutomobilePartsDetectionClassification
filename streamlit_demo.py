# Import libraries
import os, torch, sys, pickle, timm, argparse, numpy as np, streamlit as st
from glob import glob; from streamlit_free_text_select import st_free_text_select
from transformations import get_tfs  
from torchvision.datasets import ImageFolder
from utils import get_state_dict, predict, load_model, resize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
st.set_page_config(layout='wide')
sys.path.append(os.getcwd())

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameter:
    
        args   - parsed arguments, argparser object;
        
    """
    
    model_names = ["기아", "현대", "제네시스"]
    # assert args.data in model_names, "Please choose appropriate company name!"
    
    automobile_name = st_free_text_select(
        label="차량을 선택해주세요",
        options=model_names,
        index=None,
        format_func=lambda x: x.lower(),
        placeholder="선택을 위해 클릭해주세요",
        disabled=False,
        delay=300,
        label_visibility="visible")
    
    if automobile_name != None:
        st.write("선택된 차량은 ->", automobile_name)
        model_name = "genesis" if automobile_name == "제네시스" else ("kia" if automobile_name == "기아" else "hyundai")
        # Get class names for later use
        with open(f"{args.data_path}/{model_name}_cls_names.pkl", "rb") as f: cls_names = pickle.load(f)

        # Get number of classes
        num_classes = len(cls_names)
        checkpoint_path = f"ckpts/{automobile_name}_best.ckpt"
        url = "https://drive.google.com/file/d/1RaUVf1uadEHyhk-lUX4Kuaxgm3CiA4DI/view?usp=sharing" if model_name == "genesis" else ("https://drive.google.com/file/d/1m2xHGDxCG1XzKIYr00M-ugVlTAXuE7Ct/view?usp=share_link" if model_name == "kia" else "https://drive.google.com/file/d/1BUz7QKCfAOXXgtITJnmSisQxuv9txA1d/view?usp=share_link")

        # Initialize transformations to be applied
        tfs = get_tfs()[1]
        # Set a default path to the image
        default_path = glob(f"{args.root}/{automobile_name}/*.jpg")[0]

        # Load classification model
        m = load_model(args.model_name, num_classes, checkpoint_path, url = url)
        st.title(f"{automobile_name} 차량 부품 파트번호 찾는 프로그램")
        file = st.file_uploader('이미지를 업로드해주세요')

        # Get image and predicted class
        inp = file if file else default_path
        im, out = predict(m = m, path = inp, tfs = tfs, cls_names = cls_names) 
        im_tn = tfs(im)

        # Initialize GradCAM object
        cam = GradCAM(model = m, target_layers = [m.features[-1]], use_cuda = False)

        # Get a grayscale image
        grayscale_cam = cam(input_tensor = im_tn.unsqueeze(0).to("cpu"))[0, :]

        # Get visualization
        visualization = show_cam_on_image((im_tn * 255).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8) / 255, grayscale_cam, image_weight = 0.55, colormap = 2, use_rgb = True)

        st.write(f"입력된 {automobile_name} 차량 부품 이미지의 파트번호는 -> {out}입니다.")
        
        col1, col2 = st.columns(2)

        with col1: st.header("입력된 이미지:");     st.write(f"입력된 {automobile_name} 차량 부품 이미지: "); st.image(inp)
        with col2: st.header("AI 모델 성능 확인:"); st.write(f"GradCAM 결과: "); st.image(resize(Image.fromarray(visualization), im.size))
        
        
        
    else: st.write("차량명을 선택해주세요")
    
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Object Classification Demo")
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = "sample_ims", help = "Root folder for test images")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-dp", "--data_path", type = str, default = "saved_dls", help = "Dataset name")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 
