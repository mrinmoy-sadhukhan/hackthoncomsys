Hackthon Submission
========================
In this github repo there are two folder one is Task A and another is Task B. Task A is related to the code of gender classification and Task B related to Face Recognition. The details of each task such as training and testing script, evaluation script is given in below sections. 

In this repo there is `requirments.txt`file to match the python libary rquired to run the task A and task B.
This below python code helps to create virtual environment in windows to install reuirments.txt file in the virtual environment.

`python -m venv myenv`

`cd myenv\Scripts`

`activate.bat`

`pip install requirments.txt`


## Dataset Download
Dataset downloaded from hackathon website. It is unzip in `Comsys_Hackathon5` folder in this github repo. But due to limitation we can not provide full dataset here only dataset structure is given.

## Gender Classification 
Befor procced to python file pls use `unzip.py` to extract model from zip file
Task A – Gender Classification 
The evaluation Metrics are used to test the model performance are 1) Accuracy 2) Precision 3) Recall 4) F1-Score.

|Dataset| Accuracy| Precision| Recall| F1-Score|
|--------------|:----:|:------:|:-----:|:------:|
|Training set| 0.9590|0.9663 |0.9590 |0.9607 |
|Validation set |0.9360 |0.9460 |0.9360  |0.9385|
|Test set | | | | |


In the dataset training and validation data are given. and test dataset is hidden for this in table test set row is intentionally left blank. 

Before run the below script we have to open the Task_A folder. Use below command:

`cd Task_A`


To perform the training and validation a clean code is given in a python jupyter file named as :
`taskatrainscript.ipynb`

To perform testing on unseen data and generate the result of said metrics we can run `testunseen.py` file. To run the file please refer below command line code

`python testunseen.py --dataset '/path'`

To run linux environment with python 3.X.X version pls use `python3` insted of  ` python` in the command line.  
## Face Recognition

Task A – Gender Classification 
The evaluation Metrics are used to test the model performance are  1) Top-1 Accuracy 2) Macro-averaged F1-Score

|Dataset| Top-1 Accuracy| Macro-averaged F1-Score|
|--------------|:----:|:------:|
|Training set| | |
|Validation set | | |
|Test set | | |


In the dataset training and validation data are given. and test dataset is hidden for this in table test set row is intentionally left blank. 

Before run the below script we have to open the Task_B folder. Use below command:

`cd Task_B`

To perform the training and validation a clean code is given in a python jupyter file named as :
`taskbtrainscript.ipynb`

To perform testing on unseen data and generate the result of said metrics we can run `testunseen.py` file. To run the file please refer below command line code

`python testunseen.py --dataset '/path'`

To run linux environment with python 3.X.X version pls use `python3` insted of  ` python` in the command line.  

