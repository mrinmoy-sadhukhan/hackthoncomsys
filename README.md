
# 🏁 Hackathon Submission

This repository contains solutions for two tasks:

- **Task A**: Gender Classification
- **Task B**: Face Recognition

Each task is organized in its respective folder: `Task_A/` and `Task_B/`.

---

## 🔧 Environment Setup

To set up the environment and install required dependencies, follow the steps below:

1. Create a virtual environment:

   ```bash
   python -m venv myenv
   ```

2. Activate the virtual environment (Windows):

   ```bash
   cd myenv\Scripts
   activate.bat
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

> 💡 For Linux or Mac, use `source myenv/bin/activate` to activate and `python3` instead of `python`.

---

To perform testing on unseen data and generate the result of said metrics we can run `testunseen.py` file. pls change dataset path inside the file. To run the file please refer below command line code

`python testunseen.py

- After downloading, unzip the dataset into the folder `Comsys_Hackathon5/` (structure shown in repo).
- **Note**: Due to size limitations, the full dataset is not included in this repository. Only the folder structure is shown.

---

## 🧠 Task A: Gender Classification

Please refer to the `Task_A/` folder for scripts related to Gender Classification. Details about training, testing, and evaluation scripts are documented inside the folder.

### 📝 Preprocessing

Before proceeding, extract the model archive using:

```bash
python unzip.py
```

### 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

| Dataset        | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Training set   | 0.9590   | 0.9663    | 0.9590 | 0.9607   |
| Validation set | 0.9360   | 0.9460    | 0.9360 | 0.9385   |
| Test set       | *Hidden* | *Hidden*  | *Hidden* | *Hidden* |

> 🚫 The test dataset is hidden as per the competition rules, so its evaluation metrics can not be computed

### 🚀 Running the Code

1. Navigate to the Task A folder:

   ```bash
   cd Task_A
   ```

2. For training and validation:
   - Open and run the Jupyter notebook: `taskatrainscript.ipynb`

3. For testing on unseen data with accuracy, precision, recall, F1-score:
   - Update the dataset path inside `testunseen.py`
   - Run:

     ```bash
     python testunseen.py
     ```

4. For testing on unseen data with accuracy, generalization, fairness, accuracy, precision, recall, F1-score metrics
   - Update the dataset path inside `testunseenallmetricc.ipynb`
   - Open and Run the `.ipynb` file
     
> 🐧 On Linux: Use `python3` instead of `python`.

---

## 👤 Task B: Face Recognition

Please refer to the `Task_B/` folder for scripts related to face recognition. Details about training, testing, and evaluation scripts are documented inside the folder.

---

### 📊 Evaluation Metrics

- **Top-1 Accuracy**
- **Macro-averaged F1-Score**

| Dataset        | Top-1 Accuracy | Macro-averaged F1-Score |
|----------------|----------|-----------|
| Training set   | 0.8900   |   0.8900  |
| Validation set | 0.8700   |   0.8700  |
| Test set       | *Hidden* | *Hidden*  |

> 🚫 The test dataset is hidden as per the competition rules, so its evaluation metrics are not shown.

### 🚀 Running the Code

1. Navigate to the Task A folder:

   ```bash
   cd Task_B
   ```

2. For training and validation:
   - Open and run the Jupyter notebook: `taskbtrainscript.ipynb`

3. For testing on unseen data:
   - Update the dataset path inside `testunseen.py`. This script gives metrics as well as details of matching.
   - Run:

     ```bash
     python testunseen.py
     ```
     -Run: This script will give only metrics

      ```bash
     python testunseenonlymetric.py
     ```
     
> 🐧 On Linux: Use `python3` instead of `python`.

## 📎 Requirements

The file `requirements.txt` contains all necessary Python libraries to run both tasks smoothly in the virtual environment.
