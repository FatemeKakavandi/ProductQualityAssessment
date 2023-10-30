
import os
import tensorflow as tf
from src.fsutils import get_all_files_with_extension, resource_file_path
from scipy.signal import argrelextrema
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processing import load_1KHz_data, load_20KHz_data, denoising
from src.DOE_BaseLine_fcns import load_405_pr_data
import src.DOE_BaseLine_fcns
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM,SVC, NuSVC
from sklearn.model_selection import cross_validate, GridSearchCV
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf.compat.v1.disable_eager_execution()
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,confusion_matrix
from scipy.fft import fft, fftfreq
from src.CM import f1score, CM_elements, recall,precision,Accuracy,mic_f1,mac_f1
from sklearn.model_selection import KFold
from src.data_processing import interpolate_epoch
from sklearn.pipeline import make_pipeline
from src.Data_Augmentation import corr_aug_real_data


def load_baseline():
    directory = resource_file_path(
        "./resources/aio_data_BaseLine_405_2021_07_29_raw/Baseline_dataset/total_Initiator_Force")
    file_name = 'Initiator_force.csv'
    baseLine_df = pd.read_csv(os.path.join(directory, file_name))
    baseLine_df.drop(columns=baseLine_df.columns[0], axis=1, inplace=True)
    baseLine_df.dropna(axis=0, how='any', inplace=True)
    return baseLine_df


def shuffle_data(data,axis):

    return


def load_faulty_data():
    dir = resource_file_path("./resources/aio_data_DOE_Fault_Injection_2021_08_03_raw/Detectable_fault_dataset")
    file_name = 'fault_dataset'
    fault_df = pd.read_csv(os.path.join(dir, file_name))
    fault_df.drop(fault_df.columns[0], axis=1, inplace=True)
    fault_df.dropna(inplace=True)
    return fault_df


def load_norm_aug(method):
    directory = resource_file_path("./resources/Augmented_data/all_aug_data")
    file_name = str(method)+'.csv'
    aug_data = pd.read_csv(os.path.join(directory, file_name), sep=',')
    aug_data = aug_data.drop(aug_data.columns[0], axis=1)
    return aug_data


def load_abnorm_aug(method):
    directory = resource_file_path("./resources/Augmented_data/Augment_Anomaly")
    file_name = str(method) + '.csv'
    aug_data = pd.read_csv(os.path.join(directory, file_name), sep=',')
    aug_data = aug_data.drop(aug_data.columns[0], axis=1)
    aug_data.dropna(inplace=True)
    return aug_data


def l2_normalize(df):
    # l2 norm ( consider the time series as one vector
    arr = preprocessing.normalize(df,axis=0)
    out_df = pd.DataFrame(arr)
    return out_df


def zero_mean_normalize(df):
    # df -> [n_feature , n_samples]
    # but the input for the standard fcn :
    # data = [n_samples, n_feature]
    #df = df.T
    scaler = preprocessing.StandardScaler().fit(df)
    result = scaler.transform(df)
    return pd.DataFrame(result)


def OCSVM(nu):
    model = OneClassSVM(gamma='scale', nu=nu)
    return model


def OCSVM_train(X_train,model):
    model.fit(X_train)
    return model


def OCSVM_test(X_test,model):
    y_hat = model.predict(X_test)
    return y_hat


def confusion_matrix(y_true,y_hat):
    matrix = confusion_matrix(y_true,y_hat)
    return matrix


def model_results(y_true, y_hat):
    f1 = f1score(y_true,y_hat)
    recal = recall(y_true,y_hat)
    pre = precision(y_true,y_hat)
    return f1,recal, pre


def merg_fcn(df1 , df2):
    df1.interpolate(method='linear',axis=1,inplace=True)
    df2.interpolate(method='linear',axis=1,inplace=True)
    final_df = pd.merge(df1, df2, how='left', left_index=True, right_index=True)
    final_df.astype(float).interpolate(method='linear', axis=0)
    final_df.interpolate(method='linear',axis=1,inplace=True)
    return final_df


def split_fcn(df,ratio):
    num_clms = len(df.columns)
    train_clm = df.columns[0:round(num_clms*ratio)]
    test_clm = df.columns[round(num_clms*ratio):]
    train = df[train_clm]
    test = df[test_clm]
    return train, test


def df_to_array(df):
    X = df.to_numpy()
    shape = X.shape
    if shape[1] < shape[0]:
        X = X.T
    return X


def anomaly_model_without_Norm():
    # loading the data
    base_data = load_baseline()
    faulty_data = load_faulty_data()
    train_test_ratio = 0.8
    train_df, test_df = split_fcn(base_data,train_test_ratio)

    test_data = merg_fcn(test_df,faulty_data)

    X_train = df_to_array(train_df)
    y_train = [1] * len(train_df.columns)

    X_test = df_to_array(test_data)
    y_test = [1]*len(test_df.columns) + [-1]* len(faulty_data.columns)

    model = OCSVM(nu=0.04)
    model = OCSVM_train(X_train,model)
    y_hat = OCSVM_test(X_test,model)
    results = model_results(y_test, y_hat)
    return results


def anomaly_model_with_Norm(norm_method):
    # loading the Data
    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalize Data
    # Method 1
    if norm_method=='l2_norm':
        base_data = l2_normalize(base_data)
        faulty_data = l2_normalize(faulty_data)

    elif norm_method=='zero_mean':
        base_data = zero_mean_normalize(base_data)
        faulty_data = zero_mean_normalize(faulty_data)
    else:
        print('the Normalization method is not correct')

    train_test_ratio = 0.8
    train_df, test_df = split_fcn(base_data,train_test_ratio)

    test_data = merg_fcn(test_df,faulty_data)

    X_train = df_to_array(train_df)
    y_train = [1]*len(train_df.columns)

    X_test = df_to_array(test_data)
    y_test = [1]*len(test_df.columns) + [-1]* len(faulty_data.columns)

    model = OCSVM(nu=0.04)
    model = OCSVM_train(X_train,model)
    y_hat = OCSVM_test(X_test,model)
    results = model_results(y_test, y_hat)
    return results


def simple_anomaly_model_with_aug(aug_norm):
    # loading the data
    base_data = load_baseline()
    faulty_data = load_faulty_data()
    faulty_data = faulty_data.drop(columns=faulty_data.columns[5])

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)
    aug_norm = zero_mean_normalize(aug_norm)

    # train and Test Split
    train_test_ratio = 0.8
    train_df, test_df = split_fcn(base_data, train_test_ratio)

    # Merge Synthetic data and normal data
    train_real_aug = merg_fcn(train_df,aug_norm)

    # merge faulty data and real data for testing the model
    test_data = merg_fcn(test_df, faulty_data)

    X_train = df_to_array(train_real_aug)
    y_train = [1] * len(train_real_aug.columns)

    X_test = df_to_array(test_data)
    y_test = [1] * len(test_df.columns) + [-1] * len(faulty_data.columns)

    model = OCSVM(nu=0.04)
    model = OCSVM_train(X_train, model)
    y_hat = OCSVM_test(X_test, model)
    results = model_results(y_test, y_hat)
    return results


def anomaly_with_opt_aug_samp(aug_norm):
    aug_length = len(aug_norm.columns)
    f1 = []
    for i in  range(1,aug_length):
        temp_aug = aug_norm[aug_norm.columns[0:i]]
        results = simple_anomaly_model_with_aug(temp_aug)
        f1 = f1 +[results[0]]

    return f1


def split_file_to_methods(files):
    method_list = []
    for file in files:
        method = file.split('/')[-1]
        method = method.split('.')[0]
        method_list = method_list + [method]
    return method_list


def aug_list():
    directory = resource_file_path("./resources/Augmented_data/all_aug_data")
    files = get_all_files_with_extension(directory, extension='.csv')
    method_list = split_file_to_methods(files)
    return method_list


def OCSVM_anomaly(X_train,X_test,y_test):
    model = OCSVM(nu=0.04)
    model = OCSVM_train(X_train,model)
    y_hat = OCSVM_test(X_test,model)
    f1,recal , pre = model_results(y_test, y_hat)
    return f1,recal, pre


def merg_fold_au(fold,aug):
    fold_df = pd.DataFrame(fold.T)
    final_df = merg_fcn(fold_df,aug)
    arr = df_to_array(final_df)
    return arr


def anomaly_OCSVM_opt_aug_cross_val():
    # one class SVM
    # loading the data
    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)

    # Dataframe to Array
    base_arr = df_to_array(base_data)

    # split data to K folds
    n_splits = 5
    kf = KFold(n_splits)
    aug_num = 10
    # aug list
    method_list = aug_list()
    #method_list = ['permutation']
    f1_method = []
    per_df = pd.DataFrame(columns=method_list,index=range(10,90,10))
    for method in method_list:
        norm_aug = load_norm_aug(method)
        norm_aug = zero_mean_normalize(norm_aug)
        for aug_num in range(10,90,10):

            sel_norm_aug = norm_aug[norm_aug.columns[0:aug_num]]
            f1_fold = []
            faulty_aug = load_abnorm_aug(method)
            faulty_aug = zero_mean_normalize(faulty_aug)
            sel_fault_aug = faulty_aug[faulty_aug.columns[:-1]]
            sel_real_fault = faulty_data[faulty_data.columns[:-1]]
            faulty_df = merg_fcn(sel_real_fault,sel_fault_aug)

            for train_indx,val_indx in kf.split(base_arr):
                train_real_data_arr = base_arr[train_indx]
                val_real_data = base_arr[val_indx]
                X_train = merg_fold_au(train_real_data_arr,sel_norm_aug)
                X_val = merg_fold_au(val_real_data,faulty_df)
                y_val = [1]*val_real_data.shape[0]+[-1]*len(faulty_df.columns)
                f1_temp = OCSVM_anomaly(X_train, X_val, y_val)
                f1_fold = f1_fold + [f1_temp]
            per_df._set_value(aug_num,method,np.mean(f1_fold))

    return per_df

def OCSVM_cross_complete():
    # one class SVM
    # loading the data
    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)

    # Dataframe to Array
    base_train = df_to_array(base_data[base_data.columns[0:80]])
    base_test = df_to_array(base_data[base_data.columns[80:]])

    fault_val = faulty_data[faulty_data.columns[5:]]
    fault_test = faulty_data[faulty_data.columns[0:5]]
    #fault_test = faulty_data[faulty_data.columns[5:]]
    print('number of fault in validation',len(faulty_data.columns[0:5]))
    print('number of fault in test:',len(faulty_data.columns[5:]))
    # split data to K folds
    n_splits = 5
    kf = KFold(n_splits)
    # aug list
    method_list = aug_list()

    f1_method = []
    per_df = pd.DataFrame(columns=method_list, index=range(10, 90, 10))
    for method in method_list:
        norm_aug = load_norm_aug(method)
        norm_aug = zero_mean_normalize(norm_aug)
        for aug_num in range(10, 90, 10):

            sel_norm_aug = norm_aug[norm_aug.columns[0:aug_num]]
            f1_fold = []
            #faulty_aug = load_abnorm_aug(method)
            #faulty_aug = zero_mean_normalize(faulty_aug)
            #sel_fault_aug = faulty_aug[faulty_aug.columns[:-1]]

            #faulty_df = merg_fcn(sel_real_fault, sel_fault_aug)

            for train_indx, val_indx in kf.split(base_train):
                train_real_data_arr = base_train[train_indx]
                val_real_data = base_train[val_indx]
                X_train = merg_fold_au(train_real_data_arr, sel_norm_aug)
                X_val = merg_fold_au(val_real_data, fault_val)
                y_val = [1] * val_real_data.shape[0] + [-1] * len(fault_val.columns)
                f1_temp = OCSVM_anomaly(X_train, X_val, y_val)
                f1_fold = f1_fold + [f1_temp]
            per_df._set_value(aug_num, method, np.mean(f1_fold))

    return per_df

def test_ocsvm_with_aug():
    # one class SVM
    # loading the data
    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)

    # Dataframe to Array
    base_train = df_to_array(base_data[base_data.columns[0:80]])

    base_test = df_to_array(base_data[base_data.columns[80:]])

    fault_val = faulty_data[faulty_data.columns[0:5]]
    fault_test = faulty_data[faulty_data.columns[5:]]

    print('number of fault in validation',len(faulty_data.columns[0:5]))
    print('number of fault in test:',len(faulty_data.columns[5:]))
    # split data to K folds
    n_splits = 5
    kf = KFold(n_splits)
    # aug list
    method_list = ['Rand_Guided_warp', 'SPAWNER']

    f1_method = []
    for method in method_list:
        norm_aug = load_norm_aug(method)
        norm_aug = zero_mean_normalize(norm_aug)
        aug_num = 10

        sel_norm_aug = norm_aug[norm_aug.columns[0:aug_num]]

        X_train = merg_fold_au(base_train, sel_norm_aug)
        X_test = merg_fold_au(base_test, fault_test)
        y_test = [1] * base_test.shape[0] + [-1] * len(fault_test.columns)
        f1,recal, pre = OCSVM_anomaly(X_train, X_test, y_test)
        print(method)
        print('f1,recal,pre',[f1,recal,pre])

    print('No augmentation')
    f1,rec,prec = OCSVM_anomaly(base_train, X_test, y_test)
    print('f1,rec,prec',[f1,rec,prec])
    return


def store_perf(pref_df):
    directory = resource_file_path(
        "./resources/results/Aug_methods_cross_val")
    file_name = 'perf_df5.csv'
    pref_df.to_csv(os.path.join(directory, file_name),sep=';')


def stor_perf_df(pref_df,name):
    direc = resource_file_path(
        "./resources/results/OCSVM_One_anom_out")
    pref_df.to_excel(os.path.join(direc, name+'.xlsx'), float_format="%.4f")

def stor_dir_df_name(dir,name,df):
    df.to_excel(os.path.join(dir, name + '.xlsx'), float_format="%.4f")

def train_test_split(df,ratio):
    train_num  =round(ratio*len(df.columns))
    train_set = df[df.columns[:train_num]]

    test_set = df[df.columns[train_num:]]
    return train_set,test_set


def simple_binary_svm(c):
    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)
    # split data to train Test
    norm_train,norm_test = train_test_split(base_data,ratio=0.8)
    fault_train, fault_test = train_test_split(faulty_data,ratio=0.8)

    final_train_df = merg_fcn(norm_train,fault_train)
    final_test_df  = merg_fcn(norm_test,fault_test)

    # target
    y_train = [1]*len(norm_train.columns)+[-1]*len(fault_train.columns)
    y_test = [1]*len(norm_test.columns)+[-1]*len(fault_test.columns)

    # df to array
    X_train = df_to_array(final_train_df)
    X_test = df_to_array(final_test_df)

    model = SVC(kernel='rbf',C=c)
    model = model.fit(X_train,y_train)
    y_hat = model.predict(X_test)
    f1_test, recal_test, pre_test = model_results(y_test, y_hat)
    return f1_test


def simple_binary_NuSVC(nu):
    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)
    # split data to train Test
    norm_train,norm_test = train_test_split(base_data,ratio=0.6)
    fault_train, fault_test = train_test_split(faulty_data,ratio=0.7)

    final_train_df = merg_fcn(norm_train,fault_train)
    final_test_df  = merg_fcn(norm_test,fault_test)

    # target
    y_train = [1]*len(norm_train.columns)+[-1]*len(fault_train.columns)
    y_test = [1]*len(norm_test.columns)+[-1]*len(fault_test.columns)

    # df to array
    X_train = df_to_array(final_train_df)
    X_test = df_to_array(final_test_df)

    model = NuSVC(nu=nu)
    model = model.fit(X_train,y_train)
    y_hat = model.predict(X_test)
    f1_test, recal_test, pre_test = model_results(y_test, y_hat)
    return f1_test


def arr_2_df(arr,clm_list,index):
    df = pd.DataFrame(arr, columns=clm_list, index=index)
    return df



def one_anomaly_out_OCSVM_hyp_selection():

    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)
    train_sam = 62
    val_sam = 20
    ab_val_sam = 8

    # Normal data train and val and test
    base_train = df_to_array(base_data[base_data.columns[0:train_sam]])
    val_df = base_data[base_data.columns[train_sam:(train_sam+val_sam)]]
    base_test = base_data[base_data.columns[(train_sam+val_sam):]]
    Nu_list = np.linspace(0.001,0.01,19,endpoint=True)
    Nu_list = Nu_list.round(decimals=4)
    y_val = [1]*val_sam + [-1]*ab_val_sam
    y_test = [1]*len(base_test.columns)+[-1]
    column_list = [str(n)+'out' for n in range(0,9)]


    F1_val = []
    recall_val = []
    precision_val = []

    F1_test = []
    recall_test = []
    precision_test = []

    mic_f1_arr =[]
    mac_f1_arr = []
    mic_f1_arr_test = []
    mac_f1_arr_test = []

    for nu in Nu_list:
        model = OCSVM(nu)
        model.fit(base_train)
        f1_nu = []
        recal_nu = []
        pre_nu = []

        f1_test_nu = []
        recal_test_nu = []
        pre_test_nu = []

        mic_f1_nu =[]
        mac_f1_nu=[]

        mic_f1_nu_test = []
        mac_f1_nu_test = []

        i=1
        for clm in faulty_data.columns:
            # Validation part
            temp_fault_df = faulty_data.drop(columns=clm, inplace=False)
            val_set = merg_fcn(val_df, temp_fault_df)
            val_arr = df_to_array(val_set)
            y_hat = model.predict(val_arr)
            f1_val, recal_val, pre_val = model_results(y_val, y_hat)
            # Validation Micro and Macro F1
            mic_f1_nu = mic_f1_nu + [mic_f1(y_val, y_hat)]
            mac_f1_nu = mac_f1_nu + [mac_f1(y_val, y_hat)]

            f1_nu = f1_nu + [f1_val]
            recal_nu = recal_nu + [recal_val]
            pre_nu = pre_nu + [pre_val]

            # Testing time
            fault_test_df = pd.DataFrame(faulty_data[clm])
            test_set = merg_fcn(base_test,fault_test_df)
            test_arr = df_to_array(test_set)
            y_test_hat = model.predict(test_arr)
            f1_test, recal_test, pre_test = model_results(y_test,y_test_hat)

            # Test Micro and Macro F1
            mic_f1_nu_test = mic_f1_nu_test + [mic_f1(y_test,y_test_hat)]
            mac_f1_nu_test = mac_f1_nu_test + [mac_f1(y_test,y_test_hat)]


            f1_test_nu = f1_test_nu + [f1_test]
            recal_test_nu = recal_test_nu + [recal_test]
            pre_test_nu = pre_test_nu + [pre_test]

            #F1_test._set_value(nu,str(i)+'out',f1_test)
            #recall_test._set_value(nu, str(i) + 'out', recal_test)
            #precision_test._set_value(nu, str(i) + 'out', pre_test)
            i = i +1
        F1_val.append(f1_nu)
        recall_val.append(recal_nu)
        precision_val.append(pre_nu)

        F1_test.append(f1_test_nu)
        recall_test.append(recal_test_nu)
        precision_test.append((pre_test_nu))

        mic_f1_arr.append(mic_f1_nu)
        mac_f1_arr.append(mac_f1_nu)
        mic_f1_arr_test.append(mic_f1_nu_test)
        mac_f1_arr_test.append((mac_f1_nu_test))


    f1_val_df = arr_2_df(F1_val,column_list, Nu_list)
    recal_val_df =arr_2_df(recall_val,column_list, Nu_list)
    pre_val_df = arr_2_df(precision_val,column_list,Nu_list)

    f1_test_df = arr_2_df(F1_test, column_list, Nu_list)
    recal_test_df = arr_2_df(recall_test, column_list, Nu_list)
    pre_test_df = arr_2_df(precision_test, column_list, Nu_list)

    mic_f1_df = arr_2_df(mic_f1_arr,column_list,Nu_list)
    mac_f1_df = arr_2_df(mac_f1_arr, column_list, Nu_list)

    mic_f1_test_df = arr_2_df(mic_f1_arr_test,column_list,Nu_list)
    mac_f1_test_df = arr_2_df(mac_f1_arr_test,column_list,Nu_list)

    #return f1_val_df, recal_val_df,pre_val_df ,f1_test_df ,recal_test_df, pre_test_df,mic_f1_df,mac_f1_df#F1_val,recall_val,precision_val,F1_test,recall_test, precision_test
    return mic_f1_df,mac_f1_df,mic_f1_test_df,mac_f1_test_df


'''
def one_anomaly_out_OCSVM_hyp_selection():

    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)
    train_sam = 62
    val_sam = 20
    ab_val_sam = 8

    # Normal data train and val and test
    base_train = df_to_array(base_data[base_data.columns[0:train_sam]])
    val_df = base_data[base_data.columns[train_sam:(train_sam+val_sam)]]
    base_test =base_data[base_data.columns[(train_sam+val_sam):]]
    Nu_list = np.linspace(0.001,0.01,19,endpoint=True)
    Nu_list = Nu_list.round(decimals=4)
    y_val = [1]*val_sam + [-1]*ab_val_sam
    y_test = [1]*len(base_test.columns)+[-1]
    column_list = [str(n)+'out' for n in range(0,9)]


    F1_val = []
    recall_val = []
    precision_val = []

    F1_test = []
    recall_test = []
    precision_test = []
    f1_val = []
    for nu in Nu_list:
        model = OCSVM(nu)
        model.fit(base_train)
        f1_nu = []
        recal_nu = []
        pre_nu = []

        f1_test_nu = []
        recal_test_nu = []
        pre_test_nu = []

        i=1
        for clm in faulty_data.columns:
            # Validation part
            temp_fault_df = faulty_data.drop(columns=clm, inplace=False)
            val_set = merg_fcn(val_df, temp_fault_df)
            val_arr = df_to_array(val_set)
            y_hat = model.predict(val_arr)
            f1_val, recal_val, pre_val = model_results(y_val, y_hat)

            f1_nu = f1_nu + [f1_val]
            recal_nu = recal_nu + [recal_val]
            pre_nu = pre_nu + [pre_val]

            # Testing time
            fault_test_df = pd.DataFrame(faulty_data[clm])
            test_set = merg_fcn(base_test,fault_test_df)
            test_arr = df_to_array(test_set)
            y_test_hat = model.predict(test_arr)
            f1_test, recal_test, pre_test = model_results(y_test,y_test_hat)

            f1_test_nu = f1_test_nu + [f1_test]
            recal_test_nu = recal_test_nu + [recal_test]
            pre_test_nu = pre_test_nu + [pre_test]

            #F1_test._set_value(nu,str(i)+'out',f1_test)
            #recall_test._set_value(nu, str(i) + 'out', recal_test)
            #precision_test._set_value(nu, str(i) + 'out', pre_test)
            i = i +1
        F1_val.append(f1_nu)
        recall_val.append(recal_nu)
        precision_val.append(pre_nu)

        F1_test.append(f1_test_nu)
        recall_test.append(recal_test_nu)
        precision_test.append((pre_test_nu))



    f1_val_df = arr_2_df(F1_val,column_list, Nu_list)
    recal_val_df =arr_2_df(recall_val,column_list, Nu_list)
    pre_val_df = arr_2_df(precision_val,column_list,Nu_list)

    f1_test_df = arr_2_df(F1_test, column_list, Nu_list)
    recal_test_df = arr_2_df(recall_test, column_list, Nu_list)
    pre_test_df = arr_2_df(precision_test, column_list, Nu_list)

    return f1_val_df, recal_val_df,pre_val_df ,f1_test_df ,recal_test_df, pre_test_df#F1_val,recall_val,precision_val,F1_test,recall_test, precision_test

'''
def plot_data_syn():

    base_data = load_baseline()
    aug_data = load_norm_aug('Rand_Guided_warp')
    line_width = 4
    plt.plot(base_data[base_data.columns[10]], color='r',linewidth=line_width)
    plt.axis('off')
    plt.savefig('real.eps')
    plt.figure()
    plt.plot(base_data[base_data.columns[11]], color='b',linewidth=line_width)
    plt.axis('off')
    plt.savefig('real2.eps')
    plt.figure()
    plt.plot(aug_data[aug_data.columns[10]], color='g',linewidth=line_width)
    plt.axis('off')
    plt.savefig('aug.eps')


def plot_all_signals_one_product():
    directory = resource_file_path(
        "./resources/aio_data_BaseLine_800_2021_07_30_raw/All_products")
    file_name = 'DOE_BaseLine_800_20210730_005.csv'
    pr_data = pd.read_csv(os.path.join(directory, file_name))
    pr_data.drop(columns=pr_data.columns[0], axis=1, inplace=True)
    pr_data.dropna(axis=0, how='any', inplace=True)
    topics = pr_data['topic'].drop_duplicates()
    jet = plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0, 1, len(topics))))

    for topic in topics:
        topic_data = pr_data[pr_data['topic']==topic]
        sample_rate = len(topic_data['data'][topic_data.index[1]].split(','))
        if sample_rate==1 :
            plt.figure()
            plt.plot(topic_data['epoch'],topic_data['data'].astype(float).round(3), color=next(colors))
        elif sample_rate==20:
            topic_data_value = topic_data['data'].str.split(',')
            # stack topic data to make a timeseries
            topic_data_value = pd.to_numeric(topic_data_value.apply(pd.Series).stack())
            new_indx = [item[0] + 0.05 * item[1] for item in topic_data_value.index]

            epoch_prId_df = pd.DataFrame({'epoch': topic_data['epoch'],
                                          'productId': topic_data['productId']})

            topic_data_value = topic_data_value.reset_index(drop=True)
            topic_df = pd.DataFrame({'data': topic_data_value.values}, index=new_indx)
            new_topic_df = topic_df.merge(epoch_prId_df, left_index=True, right_index=True, how='left')
            new_topic_df.loc[:, ['epoch']] = interpolate_epoch(new_topic_df['epoch'])
            # new_force_df.loc[:,['epoch']] = new_force_df['epoch'].astype(float).interpolate(method='linear', axis=0, limit=None)
            # new_force_df = new_force_df.merge(pr_Id_df, left_index=True, right_index=True, how='left')
            new_topic_df.loc[:, ['productId']] = new_topic_df['productId'].bfill(axis=0)

            plt.figure()
            plt.plot(new_topic_df['epoch'],new_topic_df['data'],color=next(colors))

        #plt.title(f'signal recorded for topic:{topic}')
        #plt.axis('off')
        plt.savefig(topic+'.png',transparent=True)
    return


def plot_one_normal_data():
    data = load_baseline()
    pr_data = data[data.columns[10]]
    plt.plot(pr_data, c='#56fca2')


def one_anomaly_out_binary_NuSVM_hyp_selection():

    base_data = load_baseline()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)
    train_sam = 62
    val_sam = 20


    # Normal data train and val and test
    base_train = base_data[base_data.columns[0:train_sam]]
    val_df = base_data[base_data.columns[train_sam:(train_sam+val_sam)]]
    base_test =base_data[base_data.columns[(train_sam+val_sam):]]

    #Nu
    Nu_list = np.linspace(0.001, 0.09, 90,endpoint=True)
    Nu_list = Nu_list.round(decimals=4)

    # targets

    y_test = [1]*len(base_test.columns)+[-1]

    column_list = [str(n)+'out' for n in range(0,9)]

    # fault train
    f_train = 6
    f_val = 2

    F1_val = []
    recall_val = []
    precision_val = []

    F1_test = []
    recall_test = []
    precision_test = []

    for nu in Nu_list:
        model = NuSVC(nu)
        f1_nu = []
        recal_nu = []
        pre_nu = []

        f1_test_nu = []
        recal_test_nu = []
        pre_test_nu = []

        i=1
        for clm in faulty_data.columns:
            # faulty data
            temp_fault_df = faulty_data.drop(columns=clm, inplace=False)
            train_fault = temp_fault_df[temp_fault_df.columns[0:f_train]]
            val_fault = temp_fault_df[temp_fault_df.columns[f_train:]]

            #training part
            train_df = merg_fcn(base_train,train_fault)
            y_train = [1]*len(base_train.columns)+[-1]*len(train_fault.columns)
            train_arr = df_to_array(train_df)
            model.fit(train_arr,y_train)

            # Validation part

            val_set = merg_fcn(val_df, val_fault)
            y_val = [1]*len(val_df.columns)+[-1]*len(val_fault.columns)
            val_arr = df_to_array(val_set)

            y_hat = model.predict(val_arr)
            f1_val, recal_val, pre_val = model_results(y_val, y_hat)

            f1_nu = f1_nu + [f1_val]
            recal_nu = recal_nu + [recal_val]
            pre_nu = pre_nu + [pre_val]

            # Testing time
            fault_test_df = pd.DataFrame(faulty_data[clm])
            test_set = merg_fcn(base_test,fault_test_df)
            test_arr = df_to_array(test_set)
            y_test_hat = model.predict(test_arr)
            f1_test, recal_test, pre_test = model_results(y_test,y_test_hat)

            f1_test_nu = f1_test_nu + [f1_test]
            recal_test_nu = recal_test_nu + [recal_test]
            pre_test_nu = pre_test_nu + [pre_test]

            #F1_test._set_value(nu,str(i)+'out',f1_test)
            #recall_test._set_value(nu, str(i) + 'out', recal_test)
            #precision_test._set_value(nu, str(i) + 'out', pre_test)
            i = i +1
        F1_val.append(f1_nu)
        recall_val.append(recal_nu)
        precision_val.append(pre_nu)

        F1_test.append(f1_test_nu)
        recall_test.append(recal_test_nu)
        precision_test.append((pre_test_nu))



    f1_val_df = arr_2_df(F1_val,column_list, Nu_list)
    recal_val_df =arr_2_df(recall_val,column_list, Nu_list)
    pre_val_df = arr_2_df(precision_val,column_list,Nu_list)

    f1_test_df = arr_2_df(F1_test, column_list, Nu_list)
    recal_test_df = arr_2_df(recall_test, column_list, Nu_list)
    pre_test_df = arr_2_df(precision_test, column_list, Nu_list)

    return f1_val_df, recal_val_df,pre_val_df ,f1_test_df ,recal_test_df, pre_test_df


def load_check():
    directory = resource_file_path("./resources/Augmented_data/SPAWNER")
    file_name = 'SPAWNER2.csv'

    df = pd.read_csv(os.path.join(directory, file_name))
    return df


def merg_RGW():
    dir1 = resource_file_path("./resources/Augmented_data/all_aug_data")
    name1 = 'Rand_Guided_warp.csv'
    df1 = pd.read_csv(os.path.join(dir1,name1))
    df1.drop(columns=df1.columns[0],inplace=True)

    dir2 = resource_file_path("./resources/Augmented_data/RGW")
    name2 ='RGW2.csv'
    df2 = pd.read_csv(os.path.join(dir2,name2))
    df2.drop(columns=df2.columns[0],inplace=True)

    df = merg_fcn(df1,df2)

    df.to_csv(os.path.join(dir2,'RGW.csv'))
    return df


def merg_spawner():
    dir1 = resource_file_path("./resources/Augmented_data/all_aug_data")
    name1 = 'SPAWNER.csv'
    df1 = pd.read_csv(os.path.join(dir1,name1))
    df1.drop(columns=df1.columns[0],inplace=True)

    dir2 = resource_file_path("./resources/Augmented_data/SPAWNER")
    name2 ='SPAWNER2.csv'
    df2 = pd.read_csv(os.path.join(dir2,name2))
    df2.drop(columns=df2.columns[0],inplace=True)

    df = merg_fcn(df1,df2)

    df.to_csv(os.path.join(dir2,'SPAWNER.csv'))
    return df


def load_RGW():
    dir = resource_file_path("./resources/Augmented_data/RGW")
    name = 'RGW.csv'
    df = pd.read_csv(os.path.join(dir, name))
    return df


def one_anom_out_OCSVM_RGW_sample_sel():
    base_data = load_baseline()
    RGW_df = load_RGW()
    faulty_data = load_faulty_data()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    RGW_data = zero_mean_normalize(RGW_df)
    faulty_data = zero_mean_normalize(faulty_data)

    train_sam = 62
    val_sam = 20
    ab_val_sam = 8

    # Normal data train and val and test
    base_train = base_data[base_data.columns[0:train_sam]]
    val_df = base_data[base_data.columns[train_sam:(train_sam + val_sam)]]
    base_test = base_data[base_data.columns[(train_sam + val_sam):]]
    num_samples = range(10,200,10)

    y_val = [1] * val_sam + [-1] * ab_val_sam
    y_test = [1] * len(base_test.columns) + [-1]
    column_list = [str(n) + 'out' for n in range(0, 9)]


    F1_val = []
    recall_val = []
    precision_val = []

    F1_test = []
    recall_test = []
    precision_test = []
    nu = 0.004
    model = OCSVM(nu)

    for num in num_samples:

        f1_nu = []
        recal_nu = []
        pre_nu = []

        f1_test_nu = []
        recal_test_nu = []
        pre_test_nu = []

        i = 1
        for clm in faulty_data.columns:
            # Training
            train_RGW = RGW_data[RGW_data.columns[num-10:num]]
            #print(corr_aug_real_data(base_train, train_RGW))
            train_df = merg_fcn(base_train,train_RGW)
            train_arr = df_to_array(train_df)
            model.fit(train_arr)

            # Validation part
            temp_fault_df = faulty_data.drop(columns=clm, inplace=False)
            val_set = merg_fcn(val_df, temp_fault_df)
            val_arr = df_to_array(val_set)
            y_hat = model.predict(val_arr)
            f1_val, recal_val, pre_val = model_results(y_val, y_hat)

            f1_nu = f1_nu + [f1_val]
            recal_nu = recal_nu + [recal_val]
            pre_nu = pre_nu + [pre_val]

            # Testing time
            fault_test_df = pd.DataFrame(faulty_data[clm])
            test_set = merg_fcn(base_test, fault_test_df)
            test_arr = df_to_array(test_set)
            y_test_hat = model.predict(test_arr)
            f1_test, recal_test, pre_test = model_results(y_test, y_test_hat)

            f1_test_nu = f1_test_nu + [f1_test]
            recal_test_nu = recal_test_nu + [recal_test]
            pre_test_nu = pre_test_nu + [pre_test]

            # F1_test._set_value(nu,str(i)+'out',f1_test)
            # recall_test._set_value(nu, str(i) + 'out', recal_test)
            # precision_test._set_value(nu, str(i) + 'out', pre_test)
            i = i + 1
        F1_val.append(f1_nu)
        recall_val.append(recal_nu)
        precision_val.append(pre_nu)

        F1_test.append(f1_test_nu)
        recall_test.append(recal_test_nu)
        precision_test.append((pre_test_nu))

    f1_val_df = arr_2_df(F1_val, column_list, num_samples)
    recal_val_df = arr_2_df(recall_val, column_list, num_samples)
    pre_val_df = arr_2_df(precision_val, column_list, num_samples)

    f1_test_df = arr_2_df(F1_test, column_list, num_samples)
    recal_test_df = arr_2_df(recall_test, column_list, num_samples)
    pre_test_df = arr_2_df(precision_test, column_list, num_samples)

    return f1_val_df, recal_val_df, pre_val_df, f1_test_df, recal_test_df, pre_test_df


