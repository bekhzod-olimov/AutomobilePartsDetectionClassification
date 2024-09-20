# AutomobilePartsDetectionClassification

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify different used automobile parts (specifically, Genesis, KIA, and Hyundai). The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)). The model can be trained using two different frameworks ([PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/vitasoftAI/Recycle-Park.git
```

2. Create conda environment from yml file using the following script:

Create a virtual environment using txt file:

- Create a virtual environment:

```python
conda create -n ENV_NAME python=3.10
```

- Activate the environment using the following command:

```python
conda activate ENV_NAME
```

- Install libraries from the text file:

```python
pip install -r requirements.txt
```

3. Data Visualization

a) Genesis

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/156672f8-de64-49d0-9df5-caa606b5829a)

b) KIA

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/c10ff203-5d1a-47b5-8c28-e3828d2c4615)

c) Hyundai

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/122a346b-1d4f-4f52-9f80-c20f30e7f79a)

4. Train the AI model using the following PyTorch Lightning training script:

Train process arguments can be changed based on the following information:

![image](https://github.com/user-attachments/assets/1b15a6ea-5444-4bd7-85c1-fef6cff1a51c)

a) Genesis

```python
python train.py --data "genesis30_50" --batch_size = 64 devices = 4 --epochs 50
```

Training process progress:

![image](https://github.com/user-attachments/assets/893330dd-f905-4d9c-93ec-29913b65b362)

b) KIA

```python
python train.py --data "new_kia" --batch_size = 64 devices = 4 --epochs 50
```

Training process progress:

![image](https://github.com/user-attachments/assets/003ece4c-f4fd-4ab8-9751-47fa7eb47f20)

c) Hyundai

```python
python train.py --data "new_hyundai" --batch_size = 64 devices = 4 --epochs 50
```

Training process progress:

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/8a5c9cca-0083-4aa6-a488-80ce68414826)

5. Conduct inference using the trained model:
```python
python inference.py --data_name DATA_NAME device = "cuda:0" --lang = "eng"
```

6. Demo using pretrained AI models:

a) demo using streamlit:

```python
streamlit run streamlit_demo.py 
```

First, type of a automobile company must be choosen:
![image](https://github.com/user-attachments/assets/933bb214-9bae-4308-8c56-69aff37c3386)

b) demo using gradio:

```python
python gradio_demo.py
```

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/0143fa27-22d8-4d2f-adf6-92b13cbd826e)


7. Flask application:

a) Run the application in the terminal:

```python
python app.py
```

b) Go to the flask.py file, copy the code to jupyter notebook and run the cell.
