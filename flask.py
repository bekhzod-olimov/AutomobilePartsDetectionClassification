import pickle, timm, torch, cv2, base64, requests, json
from transformations import get_tfs
from PIL import Image
from glob import glob

root = "sample_ims/kia/"
content_type = "image/jpeg"
im_paths = glob(f"{root}*.jpg")

for idx, im_path in enumerate(im_paths):
    
    if idx == 3: break
    
    headers = {"content-type": content_type}
    _, encoded_im = cv2.imencode(".jpg", cv2.imread(im_path))
    text_im = base64.b64encode(encoded_im).decode()
    result = {}
    result["im"] = text_im
    resp = requests.post("http://localhost:8610/predict", data = json.dumps(result))
    display(Image.open(im_path))
    resp.encoding = 'ascii'
    print(resp.json()["results"])   