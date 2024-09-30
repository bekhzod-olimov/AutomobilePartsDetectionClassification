# Import libraries
import os, torch, pickle, timm, warnings, argparse, gradio as gr, numpy as np
from transformations import get_tfs; from glob import glob
from PIL import Image, ImageFont; from torchvision.datasets import ImageFolder
from torchvision import transforms as T; from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils import get_state_dict, load_model

# Turn off warnings
warnings.filterwarnings("ignore")

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameter:
    
        args   - parsed arguments, argparser object;
        
    """
    
    # Get class names for later use
    with open(f"{args.data_path}/{args.data}_cls_names.pkl", "rb") as f: cls_names = pickle.load(f)
    
    # Get number of classes
    num_classes = len(cls_names)
    model_name = "제네시스" if args.data == "genesis" else ("기아" if args.data == "kia" else "현대")
    checkpoint_path = f"ckpts/{args.data}_best.ckpt"
    url = "https://drive.google.com/file/d/1RaUVf1uadEHyhk-lUX4Kuaxgm3CiA4DI/view?usp=share_link" if args.data == "genesis" else ("https://drive.google.com/file/d/1m2xHGDxCG1XzKIYr00M-ugVlTAXuE7Ct/view?usp=share_link" if args.data == "kia" else "https://drive.google.com/file/d/1BUz7QKCfAOXXgtITJnmSisQxuv9txA1d/view?usp=share_link") 
    
    # Initialize transformations to be applied
    tfs = get_tfs((224, 224))[1]
    
    title = f"{model_name} 차량 부품 파트번호 찾는 프로그램"
    
    # Set the description
    desc = "'Click to Upload' 누르시고 이미지 선택하시거나 예시 사진 중에 고르세요!"
    
    # Get the samples to be classified
    examples = [[im] for im in glob(f"{args.root}/{args.data}/*.jpg")]
    
    # Initialize inputs with label
    inputs = gr.inputs.Image(label = f"입력된 {model_name} 차량 부품 이미지: ")
    
    # Get the model to classify the objects
    model = load_model(args.model_name, num_classes, checkpoint_path, url = url)

    def predict(inp):
        
        """
        
        This function gets an input, makes prediction and returns GradCAM visualization as well as a class name of the prediction.
        
        Parameter:
        
            inp            - input image, array.
            
        Output:
        
            visualization  - GradCAM visualization, GradCAM object;
            class_name     - class name of the prediction, str.
        
        """
    
        # Apply transformations to the image
        im = tfs(Image.fromarray(inp.astype("uint8"), "RGB"))
        
        # Initialize GradCAM object
        cam = GradCAM(model = model, target_layers = [model.features[-1]], use_cuda = False)
        
        # Get a grayscale image
        grayscale_cam = cam(input_tensor = im.unsqueeze(0).to("cpu"))[0, :]
        
        # Get visualization
        visualization = show_cam_on_image((im * 255).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8) / 255, grayscale_cam, image_weight = 0.55, colormap = 2, use_rgb = True)
        pred = torch.nn.functional.softmax(model(im.unsqueeze(0).data), dim = 1)
        vals, inds = torch.topk(pred, k = 5)
        vals, inds = vals.squeeze(0), inds.squeeze(0)
        
        out1 = f"{vals[0]:.5f} 확률로 top1 파트번호는 -> {cls_names[(inds[0].item())]}"
        out2 = f"{vals[1]:.5f} 확률로 top2 파트번호는 -> {cls_names[(inds[1].item())]}"
        out3 = f"{vals[2]:.5f} 확률로 top3 파트번호는 -> {cls_names[(inds[2].item())]}"
        out4 = f"{vals[3]:.5f} 확률로 top4 파트번호는 -> {cls_names[(inds[3].item())]}"
        out5 = f"{vals[4]:.5f} 확률로 top5 파트번호는 -> {cls_names[(inds[4].item())]}"
        
        return Image.fromarray(visualization), out1, out2, out3, out4, out5 
    
    # Initialize outputs list with gradio Image object
    outputs = [gr.outputs.Image(type = "numpy", label = "GradCAM 결과"), gr.outputs.Label(type = "numpy", label = "결과"), gr.outputs.Label(type = "numpy", label = "결과"), gr.outputs.Label(type = "numpy", label = "결과"), gr.outputs.Label(type = "numpy", label = "결과"), gr.outputs.Label(type = "numpy", label = "결과")]
    
    # Initialize gradio interface
    gr.Interface(fn = predict, inputs = inputs, outputs = outputs, title = title, description = desc, examples = examples, allow_flagging = False).launch(share = True)

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Object Classification Demo")
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = "/home/ubuntu/workspace/bekhzod/recycle_park/sample_ims", help = "Root folder for test images")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-dt", "--data", type = str, default = "genesis", help = "Dataset name")
    parser.add_argument("-dp", "--data_path", type = str, default = "saved_dls", help = "Dataset name")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args)
