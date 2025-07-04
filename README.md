
# 🏁 Hackathon Submission

This repository contains solutions for two tasks:

- **Task A**: Gender Classification
- **Task B**: Face Recognition

Each task is organized in its respective folder: `Task_A/` and `Task_B/`.

---

## Download the github repositoy

`git clone https://github.com/mrinmoy-sadhukhan/hackthoncomsys`

please use following code to enter in the downloaded repository
`cd hackthoncomsys`

## 🔧 Environment Setup

To set up the environment and install required dependencies, follow the steps below:

1. Please install anaconda software
2. Create a virtual environment:

   ```bash
   conda create -n myenv python==3.12.7
   ```

3. Activate the virtual environment (Windows):

   ```bash
   conda activate myenv
   ```

4. Install dependencies:

   ```bash
   conda env update -f environment.yml --prune
   ```

> 💡 For Linux or Mac, use `python3` instead of `python` for command line users.
---

## If you want to run train and test script on kaggle or google colab then direct upload the `.ipynb` file, it will works. Then go for `.ipynb` file from the task a and task b folder

In local machine with internet connection, perform testing on unseen data and generate the result of said metrics. We can run `testunseenallmetricc.py` file. pls change dataset path inside the file. To run the file. Please refer below command line code after activating the said environment.

`python testunseenallmetricc.py`

### **Note**: For downloading the dataset refer to hacthon webpage. After downloading, unzip the dataset into the folder `Comsys_Hackathon5/` (structure shown in repo) inside the github repository

### **Note**: Due to size limitations, the full dataset is not included in this repository. Only the folder structure is shown

---

## 🧠 Task A: Gender Classification

### Model Architecture

For Task A, we used the CoAtNet-0 architecture from the TIMM library—a hybrid of convolution 
and self-attention for capturing both local and global features. We replaced its final layer with a 
binary classifier for gender prediction. CoAtNet-0 was selected for its efficiency on small datasets 
and strong generalization, even when trained from scratch

Please refer to the `Task_A/` folder for scripts related to Gender Classification. Details about training, testing, and evaluation scripts are documented inside the folder.

#### Due to github uploading size limitation model is uploaded as zip file. In time of testing it will be automatically extracted

Link for model in google drive: [<https://drive.google.com/drive/folders/1b2xS0j3aFwVu_pvT_-FOFHMtVif10d3h?usp=sharing>](https://drive.google.com/file/d/16TfRw9X_1U4tYYZ5IGsQwNHdV1Spv0Sp/view?usp=drive_link)

### 📊 Evaluation Metrics (All data are in %)

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

| Dataset        | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Training set   | 95.90   | 96.63    | 95.90 | 96.07   |
| Validation set | 93.60   | 94.60    | 93.60 | 93.85   |
| Test set       | *Hidden* | *Hidden*  | *Hidden* | *Hidden* |

> 🚫 The test dataset is hidden as per the competition rules, so its evaluation metrics can not be computed

### 🚀 Running the Code

1. Navigate to the Task A folder:

   ```bash
   cd Task_A
   ```

2. For training and validation:
   - Open and run the Jupyter notebook: `taskatrainscript.ipynb` in vs code or other supported editor.

3. For testing on unseen data with accuracy, generalization, fairness, accuracy, precision, recall, F1-score metrics (said in webapge and email):
   - Update the dataset path inside `testunseenallmetricc.py`
   - Run:

     ```bash
     python testunseenallmetricc.py
     ```

4. For testing on unseen data with accuracy, generalization, fairness, accuracy, precision, recall, F1-score metrics (said in webapge and email) on kaggle or google colab or editor supported `.ipynb` file.
   - Update the dataset path inside `testunseenallmetricc.ipynb`
   - Run the file.

> 🐧 On Linux: Use `python3` instead of `python`.

---

## 👤 Task B: Face Recognition

### Model Architecture

For identity recognition, we implemented a Prototypical Network (ProtoNet) using ResNet-18 as the 
feature extractor, followed by a projection head that maps embeddings to a 256-dimensional latent 
space. This approach is ideal for low-shot learning with limited clean/distorted face samples per 
identity.

Please refer to the `Task_B/` folder for scripts related to face recognition and matching. Details about training, testing, and evaluation scripts are documented inside the folder.

---

### 📊 Evaluation Metrics (All data are in %)

- **Top-1 Accuracy**
- **Macro-averaged F1-Score**
- **Macro-averaged Precision**
- **Macro-avergaed Recall**
| Dataset        | Top-1 Accuracy | Macro-averaged F1-Score | Macro-averaged Precision |Macro-avergaed Recall|
|----------------|----------|-----------|------------|----------|
| Training set   | 85.98   |   85.98  | 85.98 | 85.98 |
| Validation set | 86.16  |   86.16  | 86.16 | 86.16 |
| Test set       | *Hidden* | *Hidden*  | *Hidden* | *Hidden* |

> 🚫 The test dataset is hidden as per the competition rules, so its evaluation metrics can not be computed.

### 🚀 Running the Code

1. Navigate to the Task B folder:

   ```bash
   cd Task_B
   ```

2. For training and validation:
   - Open and run the Jupyter notebook: `taskbtrainscript.ipynb`

3. For testing on unseen data:
   - Update the dataset path inside `testunseenallmetricc.py`. This script gives metrics Top-1 Accuracy, Macro-averaged F1-Score, Macro-averaged Precision and Macro-avergaed Recall as well as details of matching (said in webapge and email).
   - Run:

     ```bash
     python testunseenallmetricc.py
     ```

4. For testing on unseen data with  Top-1 Accuracy, Macro-averaged F1-Score metrics, Macro-averaged Precision and Macro-avergaed Recall as well as details of matching (said in webapge and email) on kaggle or google colab or editor supported `.ipynb` file.
   - Update the dataset path inside `testunseenallmetricc.ipynb`
   - Run the file.

> 🐧 On Linux: Use `python3` instead of `python`.

## 📎 Requirements

The file `environment.yml` contains all necessary Python libraries to run both tasks smoothly in the virtual environment.

**Note:** If you run `.ipynb` file in VS code please use kernel selector then python environments then select created enviroment.

## Technical Summary

Technical summary is given in `COMSYS_Hackathon5_Technical_Summary.pdf` file.

## :people_holding_hands: Credits

Soumya Pal and Mrinmoy Sadhukhan reserach scholar of Visva Bharati University.

