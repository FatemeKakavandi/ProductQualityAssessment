import unittest
from src.Final_Fcns import load_baseline
from source_code import anomaly_model_without_Norm,anomaly_model_with_Norm, \
    simple_anomaly_model_with_aug,load_norm_aug,anomaly_with_opt_aug_samp,\
    anomaly_OCSVM_opt_aug_cross_val, load_abnorm_aug,store_perf, load_faulty_data,\
    merg_fcn, simple_binary_svm, OCSVM_cross_complete , test_ocsvm_with_aug,\
    one_anomaly_out_OCSVM_hyp_selection,plot_data_syn, \
    plot_all_signals_one_product, plot_one_normal_data, stor_perf_df, simple_binary_NuSVC,\
    one_anomaly_out_binary_NuSVM_hyp_selection,load_check, merg_RGW, merg_spawner, one_anom_out_OCSVM_RGW_sample_sel,stor_dir_df_name
from src.fsutils import resource_file_path
from test.non_stop_testcase import CLIModeTest
import matplotlib.pyplot as plt
import numpy as np

class FinalTests(CLIModeTest):

    def test_OCSVM(self):
        # without Normalization
        # 102 normal samples
        # 11239 is the length of each normal sample
        # 9 abnormal samples
        # 11299 is the length of each abnormal sample
        # train-test ratio is 80%
        results = anomaly_model_without_Norm()
        if not self.cli_mode():
            print(results)

    def test_OCSVM_Norm1(self):
        # with Normalization -> each observation gets unit l2 norm
        # the rest is similar to the previous setting
        norm_method = 'l2_norm'
        results = anomaly_model_with_Norm(norm_method)
        if not self.cli_mode():
            print(results)

    def test_OCSVM_Norm2(self):
        # the same as before but another Normalization method
        # zero mean and std=1 for each time series
        norm_method = 'zero_mean'
        results = anomaly_model_with_Norm(norm_method)
        if not self.cli_mode():
            print(results)

    def test_OCSVM_with_aug(self):
        # OCSVM with Augmentation 100 samples
        # Method = SPAWNER
        # normalization method =  zero mean and unit std
        norm_aug_method = 'SPAWNER'
        aug_norm = load_norm_aug(norm_aug_method)
        results = simple_anomaly_model_with_aug(aug_norm)
        if not self.cli_mode():
            print(results)

    def test_OCSVM_with_optimum_aug(self):
        # OCSVM with Augmentation 100 samples
        # Method = SPAWNER
        # normalization method =  zero mean and unit std
        norm_aug_method = 'SPAWNER'
        aug_norm = load_norm_aug(norm_aug_method)
        results = anomaly_with_opt_aug_samp(aug_norm)
        plt.scatter(range(len(results)),results)
        if not self.cli_mode():
            print(results)
            plt.show()

    def test_OCSVM_Opt_Aug_with_cross_validation(self):
        perf_df = anomaly_OCSVM_opt_aug_cross_val()
        store_perf(perf_df)
        #plt.scatter(method_list,f1_method)
        '''
        if not self.cli_mode():
            print('the Augmentation methods:',method_list)
            print('the f1 performance for each method:',f1_method)
            plt.show()
        '''

    def test_Anomaly_SPAWNER_data(self):
        anomaly_data = load_abnorm_aug('SPAWNER')
        real_fault = load_faulty_data()

        final_df = merg_fcn(real_fault,anomaly_data)
        final_df.head()

    def test_binary_svm(self):
        for c in range(1,10):
            f1 = simple_binary_svm(c)
            print(f1)

    def test_complete_ocsvm(self):
        perf_df = OCSVM_cross_complete()
        store_perf(perf_df)

    def test_ocsvm_aug(self):
        test_ocsvm_with_aug()

    def test_one_anom_out(self):
        mic_f1_df,mac_f1_df,mic_f1_test,mac_f1_test  = one_anomaly_out_OCSVM_hyp_selection()
        stor_perf_df(mic_f1_df,'Mic_F1_Val')
        stor_perf_df(mac_f1_df,'Mac_F1_Val')
        stor_perf_df(mic_f1_test,'Mic_F1_Test')
        stor_perf_df(mac_f1_test,'Mac_F1_Test')

        '''
        stor_perf_df(f1_val_df, 'F1_val')
        stor_perf_df(recal_val_df, 'recall_val')
        stor_perf_df(pre_val_df,'precison_val')

        stor_perf_df(f1_test_df, 'F1_test')
        stor_perf_df(recal_test_df, 'recall_test')
        stor_perf_df(pre_test_df, 'precison_test')

        stor_perf_df(mic_f1_df, 'mic_f1')
        stor_perf_df(mac_f1_df,'mac_f1')
        #list = ['F1_val','recall_val','precision_val','F1_test','recall_test', 'precision_test']
        #list = ['F1_val', 'recall_val', 'precision_val', 'F1_test', 'recall_test', 'precision_test']
        #for pr,name in zip(perfomance, list):
        #    stor_perf_df(pr,name)
        '''

    def test_real_aug_data(self):

        plot_data_syn()
        plt.show()

    def test_plot_all_signals(self):
        plot_all_signals_one_product()
        plt.show()

    def test_plot_norm(self):
        plot_one_normal_data()
        plt.show()


    def test_one_anom_out2(self):
        f1_df = one_anomaly_out_OCSVM_hyp_selection()
        f1_df.head()
        print(f1_df)

    def test_float(self):
        for i in range():
            print(i)

    def test_NuSVC(self):
        nu_list = np.linspace(0.001,0.09,89)
        #nu_list = np.linspace(0.6,1, 40)
        f1 = []

        for nu in nu_list:
            print(f'Nu is {nu}')
            f1_result = simple_binary_NuSVC(nu)
            f1 = f1 + [f1_result]
        print(nu_list)
        print(f1)


    def test_BinaryNuSVM(self):
        f1_val_df, recal_val_df,pre_val_df, f1_test_df ,recal_test_df, pre_test_df  = one_anomaly_out_binary_NuSVM_hyp_selection()
        stor_perf_df(f1_val_df, 'F1_val')
        stor_perf_df(recal_val_df, 'recall_val')
        stor_perf_df(pre_val_df,'precison_val')

        stor_perf_df(f1_test_df, 'F1_test')
        stor_perf_df(recal_test_df, 'recall_test')
        stor_perf_df(pre_test_df, 'precison_test')
    def test_endpoint(self):
        print(np.linspace(0.001, 0.09, 90,endpoint=True))


    def test_loading_spaw_check(self):
        df = load_check()

    def test_merg_rgw(self):
        df = merg_RGW()


    def test_merg_spawner(self):
        df = merg_spawner()

    def test_samp_num(self):
        for n in range(10,200,10):
            print(n)

    def test_RGW_samp_OCSVM(self):
        dir = resource_file_path("./resources/results/OCSVM_One_anom_out_RGW")
        dfs = one_anom_out_OCSVM_RGW_sample_sel()
        names = ['f1_val_df', 'recal_val_df', 'pre_val_df', 'f1_test_df', 'recal_test_df', 'pre_test_df']
        for df,name in zip(dfs,names):
            stor_dir_df_name(dir, name, df)

if __name__ == '__main__':
    unittest.main()
