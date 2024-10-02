# Import libraries
import os, torch, argparse, pickle
from utils import get_state_dict, get_model, get_fm, create_dbs, predict_classes_new, excel_summary
from dataset import CustomDataset, InferenceCustomDataset
from transformations import get_tfs; from torch.utils.data import DataLoader

def run(args):
    
    assert args.lang in ["ko", "en"], "Please choose either English or Korean! | 영어나 한국어를 선택하세요!"
    
    tr_tfs, te_tfs = get_tfs((224, 224))
    
    ds = CustomDataset(root = os.path.dirname(args.root), data = args.data_name, lang = args.lang, transformations = te_tfs)
    cls_names, num_classes = ds.get_cls_info()
    test_ds = InferenceCustomDataset(root = args.root, data = args.data_name, transformations = te_tfs)
    print(f"There are {len(test_ds)} images to be annotated!")
    
    dl = DataLoader(dataset = ds, batch_size = 64, shuffle = True, num_workers = 8)
    inf_dl = DataLoader(dataset = test_ds, batch_size = 64, shuffle = True, num_workers = 8)
    
    if args.lang == "ko":
        print(f"테스트 데이셋에 {len(inf_dl)}개의 배치가 있습니다!")
        print(f"테스트 데이셋에 {len(cls_names)}개의 파트번호가 있습니다!")
    elif args.lang == "en":
        print(f"There are {len(inf_dl)} batches in the test dataloader!")
        print(f"There are {len(cls_names)} classes in the test dataloader!")
    
    # Set model path
    model_path = args.saved_model_path + args.data_name + "_best_model_" + args.model_name + "_new_classes.ckpt"
    
    # Load model
    model = get_model(model_name = args.model_name, n_cls = len(cls_names), lang = args.lang,
                      device = args.device, saved_model_path = model_path)
    
    qry_fms_all, im_lbls = create_dbs(model = model, test_dl = dl, model_name = args.model_name, lang = args.lang,
                                                 data_name = args.data_name, device = args.device, save_path = "train_dbs")
    
    # Clear GPU
    torch.cuda.empty_cache()
    
    # Predict classes
    predict_classes_new(model = model, qry_fms_all = qry_fms_all, im_lbls = im_lbls, test_dl = inf_dl,
                        model_name = args.model_name, data_name = args.data_name, lang = args.lang,
                        device = args.device, cls_names = cls_names, num_top_stop = 5, top_k = 5, save_path = "inf_dbs")

if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Image Classification Inference Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = "/mnt/data/bekhzod/recycle_park/inference", help = "Path to data")
    parser.add_argument("-dn", "--data_name", type = str, default = "genesis", help = "Dataset name")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-l", "--lang", type = str, default = "ko", help = "Language to be used to run the code")
    parser.add_argument("-d", "--device", type = str, default = "cuda:1", help = "GPU device name")
    parser.add_argument("-sm", "--saved_model_path", type = str, default = "saved_models/", help = "Path to the directory with the trained model")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
