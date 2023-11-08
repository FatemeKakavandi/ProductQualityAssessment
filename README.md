# Product Quality Assessment
## Introduction
In this project, We employ and compare two models, namely One class classifier and Binary classifier, using **Support Vector Machine**. The models are trained and tested on data recorded from Medical device assembly processes. The methodology and results of this project are provided in the paper entitled "Product Quality Control in Assembly Machine under Data Restricted Settings". We refer the reader to read the paper which you can access via this [link](https://ieeexplore.ieee.org/abstract/document/9976173). However the industrial data is not available, the user can employ other dataset to evaluate and compare the output of these models. In order to explain the methodology we provide an available dataset that the user can simply generate via numpy library in python. 

## Dataset
We generate normal and abnormal samples via sin function. These samples are demonstrated as bellow:
 
![alt text](https://github.com/FatemeKakavandi/ProductQualityAssessment/blob/main/data.png?raw=true)

To generate the samples the user can run the **Making_dataset.py** code. The number of normal and abnormal samples and features shape the entire dataset. Later this dataset can be used to compare the the performance of **OCSVM** and $\nu$-SVC which carry the responsibility of one class and binary classification respectively.

## Model Tuning 
For hyper-parameter tuning, we use a One-Anomaly-Out approach. The overall idea is to kick one of the anomaly samples out in the training and validation process and keep that sample for the test time. This can give us an overview of the model performance and is a validation technique for hyper-parameter selection. By averaging the values of each iteration, we can select the hyper parameter that has the highest performance in the validation process. 

In **OCSVM** the the abnormal samples are only involved in the validation and test process therefore, all of abnormal samples minus one of them are used to validate the model and one remaining sample together with test normal samples are used to test the model. However in $\nuSVC$ the abnormal samples are used in train, validation and test process. 


## References 
For further details about the performance of the models in real industrial dataset you can read the following paper:


<a id="1">[1]</a> 
F. Kakavandi, R. de Reus, C.Gomes, N. Heidari, A. Iosifidis, P.G. Larsen. (2022). 
Product Quality Control in Assembly Machine under Data Restricted Settings. 
IEEE 20th International Conference on Industrial Informatics (INDIN), 735-741.