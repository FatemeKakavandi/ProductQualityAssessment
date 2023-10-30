import unittest
import pandas as pd
from src_code import load_baseline,load_fault_set, load_sets, OCSVM ,OCSVM_train, OCSVM_test, model_results,\
    anomaly_model_without_Norm, anomaly_model_with_Norm, simple_binary_svm,simple_binary_NuSVC, \
    one_anomaly_out_OCSVM_hyp_selection,one_anomaly_out_binary_NuSVM_hyp_selection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('macosx')

class TestOCSVMandBinayClassifier(unittest.TestCase):
    def test_load_data(self):
        ## Loading data and Ploting the data
        df = load_baseline()
        plt.plot(df[df.columns[1]],df[df.columns[4]])
        plt.show()

    def test_Load_fault_data(self):
        df = load_fault_set()
        plt.plot(df[df.columns[1]],df[df.columns[4]])
        plt.show()

    def test_Load_train_test_sets(self):
        X_train,y_train,X_test,y_test = load_sets()
        print(np.shape(X_train))

    def test_train_OCSVM(self):
        results = anomaly_model_without_Norm()
        performance = ['F1 Score', 'Recall', 'Precision']
        for critera, value in zip(performance,results):
            print(f'The {critera} is : {value}')

    def test_OCSVM_norm(self):
        norm_methods =['l2_norm', 'zero_mean']
        performance = ['F1 Score', 'Recall', 'Precision']
        for norm_method in norm_methods:
            print(norm_method)
            results = anomaly_model_with_Norm(norm_method)
            for critera, value in zip(performance, results):
                print(f'The {critera} is : {value}')

    def test_SVC(self):
        f1_test = simple_binary_svm(2)
        print(f1_test)

    def test_nuSVC(self):
        nu = 0.01
        f1_test = simple_binary_NuSVC(nu)
        print(f1_test)

    def test_one_anom_out_OCSVM(self):
        restults = one_anomaly_out_OCSVM_hyp_selection()
        criteria = ['mic_f1_df','mac_f1_df','mic_f1_test_df','mac_f1_test_df']
        for elm, cr in zip(restults,criteria):
            print(f'the critera : {cr} is {elm} ')

    def test_One_Anom_out_NuSVC(self):
        restults = one_anomaly_out_binary_NuSVM_hyp_selection()




if __name__ == '__main__':
    unittest.main()
