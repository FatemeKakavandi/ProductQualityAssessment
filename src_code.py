from sklearn.svm import OneClassSVM,SVC, NuSVC
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,confusion_matrix, recall_score, precision_score

def load_baseline():
    df = pd.read_csv('./data_sources/baseline.csv')
    return df


def load_fault_set():
    df = pd.read_csv('./data_sources/fault_set.csv')
    return df


def split_fcn(df,ratio):
    num_clms = len(df.columns)
    train_clm = df.columns[0:round(num_clms*ratio)]
    test_clm = df.columns[round(num_clms*ratio):]
    train = df[train_clm]
    test = df[test_clm]
    return train, test


def merg_fcn(df1 , df2):
    df1.interpolate(method='linear',axis=1,inplace=True)
    df2.interpolate(method='linear',axis=1,inplace=True)
    final_df = pd.merge(df1, df2, how='left', left_index=True, right_index=True)
    final_df.astype(float).interpolate(method='linear', axis=0)
    final_df.interpolate(method='linear',axis=1,inplace=True)
    return final_df

def df_to_array(df):
    X = df.to_numpy()
    shape = X.shape
    if shape[1] < shape[0]:
        X = X.T
    return X


def load_sets():

    # loading normal and Abnormal samples
    # {0,1} are labels: 0 means normal and 1 means abnormal
    baseline = load_baseline()
    faulty_data = load_fault_set()

    norm_baseline = zero_mean_normalize(baseline)
    norm_faulty_data = zero_mean_normalize(faulty_data)

    train_test_ratio = 0.8
    normal_train_df, normal_test_df = split_fcn(norm_baseline, train_test_ratio)
    abnormal_train_df, abnormal_test_df = split_fcn(norm_faulty_data, train_test_ratio)

    # train set
    train_df = merg_fcn(normal_train_df, abnormal_train_df)
    y_train = [0] * len(normal_train_df.columns) + [1] * len(abnormal_train_df.columns)
    #trainy = to_categorical(y_train, num_classes=2)

    trainX = df_to_array(train_df)
    # trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))

    # test set
    test_df = merg_fcn(normal_test_df, abnormal_test_df)
    y_test = [0] * len(normal_test_df.columns) + [1] * len(abnormal_test_df.columns)
    #testy = to_categorical(y_test, num_classes=2)

    testX = df_to_array(test_df)
    # testX = testX.reshape((testX.shape[0],testX.shape[1],1))

    #trainX, trainy = shuffle(trainX, trainy)
    #testX, testy = shuffle(testX, testy)

    return trainX, y_train, testX, y_test


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
    f1 = f1_score(y_true,y_hat)
    recal = recall_score(y_true,y_hat)
    pre = precision_score(y_true,y_hat)
    return f1,recal, pre


def anomaly_model_without_Norm():
    # loading the data
    base_data = load_baseline()
    faulty_data = load_fault_set()
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
    faulty_data = load_fault_set()

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


def train_test_split(df,ratio):
    train_num  =round(ratio*len(df.columns))
    train_set = df[df.columns[:train_num]]

    test_set = df[df.columns[train_num:]]
    return train_set,test_set



def simple_binary_svm(c):
    base_data = load_baseline()
    faulty_data = load_fault_set()

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
    faulty_data = load_fault_set()

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


def mic_f1(y_true,y_pred):
    mic_f1_score = f1_score(y_true,y_pred,labels=[1,-1],average='micro')
    return mic_f1_score


def mac_f1(y_true,y_pred):
    mac_f1_score = f1_score(y_true,y_pred,labels=[1,-1],average='macro')
    return mac_f1_score


def one_anomaly_out_OCSVM_hyp_selection():

    base_data = load_baseline()
    faulty_data = load_fault_set()

    # Normalizing the data
    base_data = zero_mean_normalize(base_data)
    faulty_data = zero_mean_normalize(faulty_data)
    train_sam = 62
    val_sam = 20
    ab_val_sam = len(faulty_data.columns)-1

    # Normal data train and val and test
    base_train = df_to_array(base_data[base_data.columns[0:train_sam]])
    val_df = base_data[base_data.columns[train_sam:(train_sam+val_sam)]]
    base_test =base_data[base_data.columns[(train_sam+val_sam):]]
    Nu_list = np.linspace(0.001,0.01,19,endpoint=True)
    Nu_list = Nu_list.round(decimals=4)
    y_val = [1]*val_sam + [-1]*ab_val_sam
    y_test = [1]*len(base_test.columns)+[-1]
    column_list = [str(n)+'out' for n in range(0,len(faulty_data.columns))]


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


    #f1_val_df = arr_2_df(F1_val,column_list, Nu_list)
    #recal_val_df =arr_2_df(recall_val,column_list, Nu_list)
    #pre_val_df = arr_2_df(precision_val,column_list,Nu_list)

    #f1_test_df = arr_2_df(F1_test, column_list, Nu_list)
    #recal_test_df = arr_2_df(recall_test, column_list, Nu_list)
    #pre_test_df = arr_2_df(precision_test, column_list, Nu_list)

    mic_f1_df = arr_2_df(mic_f1_arr,column_list,Nu_list)
    mac_f1_df = arr_2_df(mac_f1_arr, column_list, Nu_list)

    mic_f1_test_df = arr_2_df(mic_f1_arr_test,column_list,Nu_list)
    mac_f1_test_df = arr_2_df(mac_f1_arr_test,column_list,Nu_list)

    #return f1_val_df, recal_val_df,pre_val_df ,f1_test_df ,recal_test_df, pre_test_df,mic_f1_df,mac_f1_df#F1_val,recall_val,precision_val,F1_test,recall_test, precision_test
    return mic_f1_df,mac_f1_df,mic_f1_test_df,mac_f1_test_df

def one_anomaly_out_binary_NuSVM_hyp_selection():

    base_data = load_baseline()
    faulty_data = load_fault_set()

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

    column_list = [str(n)+'out' for n in range(0,len(faulty_data.columns))]

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
        model = NuSVC(nu=nu)
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

