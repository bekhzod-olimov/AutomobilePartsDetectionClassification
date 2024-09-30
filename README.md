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

![image](https://github.com/user-attachments/assets/72fbcb5f-7dec-4e01-866d-87ec5337176e)

b) KIA

![image](https://github.com/user-attachments/assets/989e859e-431a-4181-b4a3-9968d74d673e)

c) Hyundai

![image](https://github.com/user-attachments/assets/a67229c9-0d31-4cca-82c8-ad0c6b1726eb)

4. Train the AI model using the following PyTorch Lightning training script:

Train process arguments can be changed based on the following information:

![image](https://github.com/user-attachments/assets/1b15a6ea-5444-4bd7-85c1-fef6cff1a51c)

a) Genesis

```python
python train.py --data "genesis30_50" --batch_size = 64 devices = 4 --epochs 50
```

Training process progress:

![image](https://github.com/user-attachments/assets/893330dd-f905-4d9c-93ec-29913b65b362)

Learning Curves:

![image](https://github.com/user-attachments/assets/ef9ea0d2-956b-4a48-8391-ef6f1d270121)

![image](https://github.com/user-attachments/assets/166f4ad4-0ed5-4e2d-bad7-4dcb6918bac3)

b) KIA

```python
python train.py --data "new_kia" --batch_size = 64 devices = 4 --epochs 50
```

Training process progress:

![image](https://github.com/user-attachments/assets/003ece4c-f4fd-4ab8-9751-47fa7eb47f20)

Learning Curves:

![image](https://github.com/user-attachments/assets/82a2f592-88c4-4bf1-adf3-996874632925)
![image](https://github.com/user-attachments/assets/e6259ed3-3c64-43ad-b227-5aaacb44949c)

c) Hyundai

```python
python train.py --data "new_hyundai" --batch_size = 64 devices = 4 --epochs 50
```

Training process progress:

![image](https://github.com/user-attachments/assets/84851bee-1519-4a4b-a6d3-4954f7d877d7)

Learning Curves:

![image](https://github.com/user-attachments/assets/da4d9b94-5d28-40b3-bde6-cdf7b710f768)
![image](https://github.com/user-attachments/assets/2c45df08-5d20-4fb4-bd97-c59277f277f2)


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

<img width="1522" alt="gradio_genesis" src="https://github.com/user-attachments/assets/bb706fa5-161a-4712-8e41-3565e20a8329">


 *** [Genesis](https://github.com/bekhzod-olimov/AutomobilePartsDetectionClassification/blob/main/screenshots/gradio_genesis.png)
 *** [Hyundai](https://github.com/bekhzod-olimov/AutomobilePartsDetectionClassification/blob/main/screenshots/gradio_hyundai.png)
 *** [KIA](https://github.com/bekhzod-olimov/AutomobilePartsDetectionClassification/blob/main/screenshots/gradio_kia.png)

7. Flask application:

a) Run the application in the terminal:

```python
python app.py
```

b) Go to the flask.py file, copy the code to jupyter notebook and run the cell.
