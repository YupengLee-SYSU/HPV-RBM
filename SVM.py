from sklearn.svm import SVC
import numpy as np
from data import Data
from roc import ROC
import BaseOperation as BO
from sklearn.externals import joblib
import math


class HpvSVM:
    def __init__(self, train_data, train_label, valid_data=None, valid_label=None):
        self.train_data = train_data
        self.sample_num = self.train_data.shape[0]
        self.train_label = np.array(train_label)
        self.valid_data = valid_data
        self.valid_label = np.array(valid_label)
        self.model = SVC(probability=True, verbose=True)
        self.train_score, self.valid_score = None, None

    def run(self):
        self.model.fit(self.train_data, self.train_label)
        self.train_score = self.model.predict_proba(self.train_data)[:, 1]
        # self.train_score = self.model.predict(self.train_data)
        self.valid_score = self.model.predict_proba(self.valid_data)[:, 1]

    def eval(self):
        train_roc_eval = ROC(result=self.train_score, label=self.train_label)
        valid_roc_eval = ROC(result=self.valid_score, label=self.valid_label)
        print('training AUC:'+str(train_roc_eval.auc))
        print('validating AUC:'+str(valid_roc_eval.auc))

    def outputPredicts(self):
        result = None
        if self.valid_data is not None:
            result = self.model.predict_proba(self.valid_data)[:, 1]
        return result

    def save_valid_results(self, saveBase, repeat_n=1):
        self.save_result = self.outputPredicts()
        self.save_label = self.valid_label
        BO.writeList2txt(saveBase+'independ\\repeat_'+str(repeat_n)+'\\result.txt', self.save_result)
        BO.writeList2txt(saveBase+'independ\\repeat_'+str(repeat_n)+'\\label.txt', self.save_label)
        print('finished saving ... ')

    def shuffle(self, data, label):
        index = np.arange(self.sample_num)
        np.random.shuffle(index)
        sf_data = np.zeros(np.shape(data))
        sf_label = np.zeros([np.shape(label)[0], 1])
        for tmp_index in range(self.sample_num):
            sf_data[tmp_index, :] = data[index[tmp_index], :]
            sf_label[tmp_index, 0] = label[index[tmp_index]]
        return sf_data, sf_label

    def split_data(self, data, label, N):
        data_set, label_set = [], []
        fold_size = math.floor(self.sample_num/N)
        for fold_count in range(N):
            Start = fold_count*fold_size
            End = Start+fold_size-1
            tmp_fold_size = fold_size
            if fold_count == N-1:
                End = self.sample_num-1
                tmp_fold_size = self.sample_num-(N-1)*fold_size

            data_block_valid, label_block_valid = data[Start:End+1, :], label[Start:End+1]
            data_block_train, label_block_train = np.zeros([self.sample_num-tmp_fold_size, data.shape[1]]), np.zeros([self.sample_num-tmp_fold_size, 1])
            if fold_count == 0:
                data_block_train, label_block_train = data[End+1:self.sample_num, :], label[End+1:self.sample_num]
            else:
                if fold_count == N-1:
                    data_block_train, label_block_train = data[0:Start, :], label[0:Start, :]
                else:
                    data_block_train[0:Start, :], label_block_train[0:Start] = data[0:Start, :], label[0:Start]
                    data_block_train[Start:self.sample_num-tmp_fold_size, :], label_block_train[Start:self.sample_num-tmp_fold_size, :] = data[End+1:self.sample_num, :], label[End+1:self.sample_num, :]

            data_set.append({'train': data_block_train, 'valid': data_block_valid})
            label_set.append({'train': label_block_train, 'valid': label_block_valid})
        return data_set, label_set

    def cross_val(self, n_fold, repeat_num=1, saveBase=None):
        sf_data, sf_label = self.shuffle(data=self.train_data, label=self.train_label)
        data_set, label_set = self.split_data(sf_data, sf_label, n_fold)
        result_list, label_list = np.zeros([self.sample_num, 1]), np.zeros([self.sample_num, 1])
        index = 0
        for i in range(n_fold):
            print("processing fold"+str(i+1))
            self.train_data, self.train_label = data_set[i]['train'], label_set[i]['train']
            self.valid_data, self.valid_label = data_set[i]['valid'], label_set[i]['valid']
            fold_size = self.valid_data.shape[0]
            self.model = SVC(probability=True, verbose=False)
            self.run()
            self.eval()
            valid_result, valid_label = self.outputPredicts(), self.valid_label
            fold_ROC = ROC(result=valid_result, label=valid_label)
            print("======================================= AUC:"+str(fold_ROC.auc))
            result_list[index:index+fold_size, 0] = valid_result
            label_list[index:index+fold_size, :] = valid_label
            index = index+fold_size

        GlobalROC = ROC(result=result_list, label=label_list)
        print('*************************************** global AUC:'+str(GlobalROC.auc))

        if saveBase is not None:
            self.save_result = result_list
            self.save_label = label_list
            saveDir = saveBase+str(n_fold)+'_fold\\repeat_'+str(repeat_num)
            BO.writeResult2txt(saveDir+'\\result.txt', self.save_result)
            BO.writeResult2txt(saveDir+'\\label.txt', self.save_label)
            print('finished saving ...')


if __name__ == '__main__':

    randomforest_n = 800
    isLinux = True

    # baseDir = '/home/liyupeng/HPV-RBM/'
    baseDir = 'F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\'

    dataDir = baseDir + BO.windows2linuxDirs('DATA2\\train\\', isLinux)
    train_matrix_file = dataDir + 'train_matrix_' + str(randomforest_n) + '.txt'
    train_label_file = dataDir + 'train_label.txt'
    train_data = Data(matrix_file=train_matrix_file, label_file=train_label_file)

    validDir = baseDir + BO.windows2linuxDirs('DATA2\\valid\\', isLinux)
    valid_matrix_file = validDir + 'valid_matrix_' + str(randomforest_n) + '.txt'
    valid_label_file = validDir + 'valid_label.txt'
    valid_data = Data(matrix_file=valid_matrix_file, label_file=valid_label_file)

    trained_scalar_normer = joblib.load(baseDir + BO.windows2linuxDirs('norm_model\\StanderScalor.model', isLinux))
    train_matrix_norm = trained_scalar_normer.transform(train_data.matrix)
    valid_matrix_norm = trained_scalar_normer.transform(valid_data.matrix)

    for repeat in range(10):
        hpv_new = HpvSVM(train_data=train_matrix_norm, train_label=train_data.label_list,
                         valid_data=valid_matrix_norm, valid_label=valid_data.label_list)
        hpv_new.run()
        hpv_new.eval()
        hpv_new.save_valid_results(saveBase=baseDir+'SVM\\', repeat_n=repeat+1)
        # hpv_new.cross_val(n_fold=fold_n, repeat_num=repeat+1, saveBase=baseDir + 'SVM\\')

