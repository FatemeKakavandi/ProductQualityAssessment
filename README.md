# Product Quality Assessment
In this project, We employ and compare two models, namely One class classifier and Binary classifier, using **Support Vector Machine**. The models are trained and tested on data recorded from Medical device assembly processes. The methodology and results of this project are provided in the paper entitled "Product Quality Control in Assembly Machine under Data Restricted Settings". We refer the reader to read the paper which you can access via this [link](https://ieeexplore.ieee.org/abstract/document/9976173). However the industrial data is not available, the user can employ other dataset to evaluate and compare the output of these models. In order to explain the methodology we provide an available dataset that the user can simply generate via numpy library in python. 

We generate normal and abnormal samples via sin function. These samples are demonstrated as bellow:
 
![alt text](https://github.com/FatemeKakavandi/ProductQualityAssessment/blob/main/data.png?raw=true)

To generate the samples the user can run the **Making_dataset.py** code. The number of normal and abnormal samples and features shape the entire dataset. Later this dataset can be used to compare the **OCSVM** and $\nu$SVC.

For hyperparameter tuning, we use a One-Anomaly-Out approach.