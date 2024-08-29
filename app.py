# Import libraries
import io, json, timm, base64, torch, argparse, pickle
from torchvision import models
from PIL import Image; from utils import load_model
from flask import Flask, jsonify, request
from transformations import get_tfs

# Initialize application
app = Flask(__name__)

def run(args):

    # Get class names
    with open(f"saved_dls/{args.data}_cls_names.pkl", "rb") as f: cls_names = pickle.load(f)
    # Get transformations to be applied
    tfs = get_tfs((224, 224))[1]
    # Set the urls to download pretrained weights for every model
    url = "https://drive.google.com/file/d/1RaUVf1uadEHyhk-lUX4Kuaxgm3CiA4DI/view?usp=sharing" if args.data == "genesis" else ("https://drive.google.com/file/d/1m2xHGDxCG1XzKIYr00M-ugVlTAXuE7Ct/view?usp=share_link" if args.data == "kia" else "https://drive.google.com/file/d/1BUz7QKCfAOXXgtITJnmSisQxuv9txA1d/view?usp=share_link")
    # Load AI model
    model = load_model(model_name = args.model_name, num_classes = len(cls_names), checkpoint_path = f"ckpts/{args.data}_best.ckpt", url = url)

    @app.route('/predict', methods=['POST'])
    def predict():

        """
        This functon classifies an input image and converts the output into json format.

        Output:

             json    - jsonified prediction of the AI model.
        
        """

        # Load image using json
        im = json.loads(request.data)['im']
        # Get original image
        jpg_original = base64.b64decode(im)
        # Get AI models output
        results = get_prediction(model = model, cls_names = cls_names, image_bytes=jpg_original)

        return jsonify({"results": results})

    def get_prediction(model, image_bytes, cls_names):

        """

        This function gets several arguments and returns prediction results.

        Arguments:

             model      - an AI model;
             
        
        """

        results = {}
        im = tfs(Image.open(io.BytesIO(image_bytes)))
        pred = torch.nn.functional.softmax(model(im.unsqueeze(0).data), dim = 1)

        vals, inds = torch.topk(pred, k = 5)
        vals, inds = vals.squeeze(0), inds.squeeze(0)

        for idx, (val, ind) in enumerate(zip(vals, inds)):
            results[f"top_{idx + 1}"] = cls_names[ind.item()]

        return results

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Object Classification Demo")
    
    # Add arguments
    parser.add_argument("-dt", "--data", type = str, default = "kia", help = "Dataset name")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args)
    
    app.run(host='0.0.0.0', debug=False, port=8610)
    
