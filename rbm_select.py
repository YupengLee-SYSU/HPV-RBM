import BaseOperation as BO
import numpy as np
import math
from MLP import YpMLP
from roc import ROC


class RBM_select:
    def __init__(self, data, label, gene_list, repeat_n, params, saveBase):
        self.saveBase = saveBase
        self.all_data = data
        self.sample_num = np.shape(self.all_data)[0]
        self.dim = np.shape(self.all_data)[1]
        self.all_label = label
        self.gene_list = gene_list
        self.repeat_n = repeat_n
        self.params = params
        self.valid_num = int(math.floor(self.sample_num / 4))
        self.train_num = self.sample_num - self.valid_num

    def gen_sorted_list(self, repeat_num=1):
        Dir = self.saveBase + 'repeat_' + str(repeat_num) + '\\'
        coef_list = np.array(BO.load_list(Dir+'coef_auc.txt'))
        newGeneList = []
        index = np.argsort(coef_list)
        for i in range(len(self.gene_list)):
            newGeneList.append(self.gene_list[index[i]])

        BO.save_info_list(Dir+'sorted_gene_list.txt', newGeneList)
        # return newGeneList

    def run(self):
        for i in range(self.repeat_n):
        # for i in range(self.repeat_n):
            print('processing for repeat:'+str(i+1))
            index = np.arange(self.sample_num)
            np.random.shuffle(index)

            valid_data = np.zeros([self.valid_num, self.dim])
            valid_label = []
            for valid_i in range(self.valid_num):
                valid_data[valid_i, :] = self.all_data[index[valid_i], :]
                valid_label.append(self.all_label[index[valid_i]])

            train_data = np.zeros([self.train_num, self.dim])
            train_label = []
            for train_i in range(self.valid_num, self.sample_num):
                train_data[train_i-self.valid_num, :] = self.all_data[index[train_i], :]
                train_label.append(self.all_label[index[train_i]])

            rbm_model = YpMLP(train_data=train_data, train_label=train_label, params=self.params,
                              valid_data=valid_data, valid_label=valid_label)
            rbm_model.run_trainer(Epoch_Num=500, Batch_Size=10)
            tmp_save_dir = self.saveBase+'repeat_'+str(i+1)+'\\'
            BO.createDir(tmp_save_dir)
            rbm_model.save_results4select(saveBase=tmp_save_dir)

            out_matrix = np.zeros([self.valid_num, len(self.gene_list)])
            sub_valid_label = None
            for j in range(len(self.gene_list)):
                # print('--------> gene:'+str(j+1))
                sub_valid_data = valid_data
                sub_valid_data[:, j] = 0
                sub_valid_output = rbm_model.outputPred4Change(my_valid_data=sub_valid_data)
                sub_valid_label = rbm_model.valid_label[:, 1]
                out_matrix[:, j] = sub_valid_output
            BO.save_pure_matrix(tmp_save_dir+'result_matrix.txt', out_matrix)
            BO.writeList2txt(tmp_save_dir+'sel_label.txt', sub_valid_label)

    def get_sorted_gene(self, sel='auc', repeat_num=1):
        Dir = self.saveBase+'repeat_'+str(repeat_num)+'\\'
        data_matrix = BO.load_np_matrix(Dir+'result_matrix.txt', self.valid_num, len(self.gene_list))
        label = BO.load_np_label(Dir+'sel_label.txt', self.valid_num)
        result = BO.load_np_label(Dir+'result.txt', self.valid_num)
        if sel == 'auc':
            std_ROC = ROC(result=result, label=label)
            std_auc = std_ROC.auc
            coef = []
            for i in range(len(self.gene_list)):
                sub_roc = ROC(result=data_matrix[:, i], label=label)
                sub_auc = sub_roc.auc
                coef.append((sub_auc-std_auc)/std_auc)
                if (i+1) % 50 == 0:
                    print('gene number:'+str(i+1))
            BO.save_info_list(Dir+'coef_auc.txt', coef)

        if sel == 'Sn':
            std_ROC = ROC(result=result, label=label)
            std_sn = std_ROC.default_Sn
            coef = []
            for i in range(len(self.gene_list)):
                sub_roc = ROC(result=data_matrix[:, i], label=label)
                sub_sn = sub_roc.default_Sn
                coef.append((sub_sn - std_sn) / std_sn)
            BO.save_info_list(Dir + 'coef_sn.txt', coef)





