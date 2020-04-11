import tensorflow as tf
from roc import ROC
from dataIterator import DataIterator
import numpy as np
import math
import BaseOperation as BO


class YpMLP:
    def __init__(self, train_data, train_label, n_layers=None, n_neurons=None, params=None,
                 valid_data=None, valid_label=None, isSave=False, isSaveModelParams=False):
        self.weight_list, self.bias_list = {}, {}

        self.epoch_train_err, self.epoch_valid_err = [], []
        self.save_result, self.save_label = None, None
        self.isSave = isSave
        self.train_data = train_data
        self.sample_num = self.train_data.shape[0]
        self.train_label_list = train_label
        self.train_label = self.expand_labels(label_list=self.train_label_list)

        self.valid_data = valid_data
        self.valid_label = self.expand_labels(label_list=valid_label)

        self.isRegularize = False
        self.beta = 0.0001
        self.regularizer = 0

        self.params = params
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.xs, self.ys = None, None
        self.lossFunc, self.trainer = None, None
        self.model = None
        if self.params is not None:
            if isSaveModelParams is True:
                self.config_by_params_savable()
            else:
                self.config_by_params()
        else:
            self.config_nets()
        self.config_trainer()

        self.sess = tf.Session()

    def expand_labels(self, label_list):
        label_arr = np.zeros([len(label_list), 2])
        for i in range(len(label_list)):
            if label_list[i] == 1:
                label_arr[i, :] = np.array([0, 1])
            if label_list[i] == 0:
                label_arr[i, :] = np.array([1, 0])
        return label_arr

    def outputPredicts(self):
        predResults = None
        if self.valid_data is not None:
            predResults = self.sess.run(self.model, feed_dict={self.xs: self.valid_data})
        return predResults[:, 1]

    def outputPred4Change(self, my_valid_data):
        predResults = self.sess.run(self.model, feed_dict={self.xs: my_valid_data})
        return predResults[:, 1]

    def run_trainer(self, Epoch_Num, Batch_Size):
        Init = tf.global_variables_initializer()
        self.sess.run(Init)
        # -------- train ----------------------------------------------------
        epoch_num = Epoch_Num
        batch_size = Batch_Size
        data_iter = DataIterator(matrix_data=self.train_data, label_data=self.train_label, batchSize=batch_size)

        for i in range(epoch_num):
            while data_iter.isHasNext:
                tmp_data, tmp_label = data_iter.next_batch()
                self.sess.run(self.trainer, feed_dict={self.xs: tmp_data, self.ys: tmp_label})
            tmp_err = self.sess.run(self.lossFunc, feed_dict={self.xs: self.train_data, self.ys: self.train_label})

            if i % 2 == 0:
                epoch_predict = self.sess.run(self.model, feed_dict={self.xs: self.train_data})
                epoch_ROC = ROC(result=epoch_predict[:, 1], label=self.train_label[:, 1])

                epoch_valid = self.sess.run(self.model, feed_dict={self.xs: self.valid_data})
                epoch_ROC_valid = ROC(result=epoch_valid[:, 1], label=self.valid_label[:, 1])
                print('epoch: ' + str(i+1) + ', loss: ' + str(tmp_err))
                print('==========> current training AUC:' + str(epoch_ROC.auc)+', validating AUC:'+str(epoch_ROC_valid.auc))

                self.epoch_train_err.append(epoch_ROC.auc)
                self.epoch_valid_err.append(epoch_ROC_valid.auc)
            data_iter.shuffle_data()

    def save_valid_results(self, saveBase, repeat_n=1):
        self.save_result = self.outputPredicts()
        self.save_label = self.valid_label[:, 1]
        BO.writeList2txt(saveBase+'independ\\repeat_'+str(repeat_n)+'\\result.txt', self.save_result)
        BO.writeList2txt(saveBase+'independ\\repeat_'+str(repeat_n)+'\\label.txt', self.save_label)
        print('finished saving ... ')

    def save_results4select(self, saveBase):
        self.save_result = self.outputPredicts()
        self.save_label = self.valid_label[:, 1]
        BO.writeList2txt(saveBase+'result.txt', self.save_result)
        BO.writeList2txt(saveBase+'label.txt', self.save_label)
        print('finished saving ... ')

    def config_trainer(self):
        self.lossFunc = -tf.reduce_mean(self.ys * tf.log(self.model))
        if self.isRegularize is True:
            self.lossFunc = self.lossFunc+self.beta*self.regularizer
        self.trainer = tf.train.GradientDescentOptimizer(0.005).minimize(self.lossFunc)

    def config_by_params(self):
        weights = self.params['weights']
        bias = self.params['bias']
        self.n_layers = len(weights)+1
        inSize = weights[0].shape[0]
        self.xs = tf.placeholder(tf.float32, [None, inSize])
        self.ys = tf.placeholder(tf.float32, [None, 2])
        in_elem = self.xs
        for n in range(len(weights)):
            tmp_block = self.DenseLayer(in_data=in_elem, in_size=weights[n].shape[0], out_size=weights[0].shape[1]
                                    , active_func=tf.nn.sigmoid, isLinear=False, w_b={'w': weights[n], 'b': bias[n]})
            in_elem = tmp_block
        out_layer = self.DenseLayer(in_data=in_elem, in_size=weights[len(weights)-1].shape[1], out_size=2
                                        , active_func=tf.nn.sigmoid, isLinear=False)
        self.model = tf.nn.softmax(out_layer)

    def config_by_params_savable(self):
        weights = self.params['weights']
        bias = self.params['bias']
        self.n_layers = len(weights)+1

        print(self.n_neurons)

        for n in range(len(self.n_neurons)-2):
            tmp_in_size = self.n_neurons[n]
            tmp_out_size = self.n_neurons[n+1]
            self.weight_list['layer_' + str(n)] = tf.Variable(tf.constant(weights[n]),
                                                    name='layer_' + str(n) + '_weight')
            self.bias_list['layer_' + str(n)] = tf.Variable(tf.constant(bias[n]), name='layer_' + str(n) + '_bias')
        last_n = len(self.n_neurons)-2
        self.weight_list['layer_' + str(last_n)] = tf.Variable(tf.random_normal([self.n_neurons[last_n], self.n_neurons[last_n+1]]),
                                                name='layer_' + str(last_n) + '_weight')
        self.bias_list['layer_' + str(last_n)] = tf.Variable(tf.zeros([self.n_neurons[last_n+1]]) + 0.001,
                                                             name='layer_' + str(last_n) + '_bias')

        print('weight keys:')
        print(self.weight_list.keys())
        print('bias keys:')
        print(self.bias_list.keys())

        self.xs = tf.placeholder(tf.float32, [None, self.n_neurons[0]])
        self.ys = tf.placeholder(tf.float32, [None, 2])
        in_elem = self.xs
        for m in range(len(self.n_neurons)-1):
            tmp_block = self.DenseLayer_savable(in_data=in_elem, W=self.weight_list, B=self.bias_list,
                                                active_func=tf.nn.sigmoid, isLinear=False, layer_count=m)
            in_elem = tmp_block
        self.model = tf.nn.softmax(in_elem)

    def config_nets(self):
        self.xs = tf.placeholder(tf.float32, [None, self.n_neurons[0]])
        self.ys = tf.placeholder(tf.float32, [None, 2])
        in_elem = self.xs
        for n in range(self.n_layers):
            tmp_in_size = self.n_neurons[n]
            tmp_out_size = self.n_neurons[n+1]
            tmp_block = self.DenseLayer(in_data=in_elem, in_size=tmp_in_size, out_size=tmp_out_size
                                        , active_func=tf.nn.sigmoid, isLinear=False)
            in_elem = tmp_block
        self.model = tf.nn.softmax(in_elem)

    def DenseLayer(self, in_data, in_size, out_size, active_func, isLinear=False, w_b=None):
        if w_b is None:
            weights = tf.Variable(tf.random_normal([in_size, out_size]))
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.001)
        else:
            w_value, b_value = w_b['w'], w_b['b']
            weights = tf.Variable(tf.constant(w_value))
            bias = tf.Variable(tf.constant(b_value))

        # in_data = tf.nn.dropout(in_data, keep_prob=0.8)
        self.regularizer = self.regularizer+tf.nn.l2_loss(weights)+tf.nn.l2_loss(bias)
        hidden = tf.matmul(in_data, weights) + bias
        if isLinear is False:
            out_data = active_func(hidden)
        else:
            out_data = hidden
        return out_data

    def DenseLayer_savable(self, in_data, W, B, active_func, isLinear=False, layer_count=0):

        weights = W['layer_'+str(layer_count)]
        bias = B['layer_'+str(layer_count)]

        hidden = tf.matmul(in_data, weights) + bias
        if isLinear is False:
            out_data = active_func(hidden)
        else:
            out_data = hidden
        return out_data

    def shuffle(self, data, label):
        index = np.arange(self.sample_num)
        np.random.shuffle(index)
        sf_data = np.zeros(np.shape(data))
        sf_label = []
        for tmp_index in range(self.sample_num):
            sf_data[tmp_index, :] = data[index[tmp_index], :]
            sf_label.append(label[index[tmp_index]])
        return sf_data, self.expand_labels(sf_label)

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

            data_block_valid, label_block_valid = data[Start:End+1, :], label[Start:End+1, :]
            data_block_train, label_block_train = np.zeros([self.sample_num-tmp_fold_size, data.shape[1]]), np.zeros([self.sample_num-tmp_fold_size, label.shape[1]])
            if fold_count == 0:
                data_block_train, label_block_train = data[End+1:self.sample_num, :], label[End+1:self.sample_num]
            else:
                if fold_count == N-1:
                    data_block_train, label_block_train = data[0:Start, :], label[0:Start, :]
                else:
                    data_block_train[0:Start, :], label_block_train[0:Start, :] = data[0:Start, :], label[0:Start, :]
                    data_block_train[Start:self.sample_num-tmp_fold_size, :], label_block_train[Start:self.sample_num-tmp_fold_size, :] = data[End+1:self.sample_num, :], label[End+1:self.sample_num, :]

            data_set.append({'train': data_block_train, 'valid': data_block_valid})
            label_set.append({'train': label_block_train, 'valid': label_block_valid})
        return data_set, label_set

    def cross_val(self, n_fold, repeat_num=1, saveBase=None):
        sf_data, sf_label = self.shuffle(data=self.train_data, label=self.train_label_list)
        data_set, label_set = self.split_data(sf_data, sf_label, n_fold)
        result_list, label_list = np.zeros([self.sample_num, 1]), np.zeros([self.sample_num, 1])
        index = 0
        for i in range(n_fold):
            print("processing fold"+str(i+1))
            if self.params is not None:
                self.config_by_params()
            else:
                self.config_nets()
            self.config_trainer()

            self.train_data, self.train_label = data_set[i]['train'], label_set[i]['train']
            self.valid_data, self.valid_label = data_set[i]['valid'], label_set[i]['valid']
            fold_size = self.valid_data.shape[0]

            self.run_trainer(Epoch_Num=400, Batch_Size=10)

            valid_result, valid_label = self.outputPredicts(), self.valid_label
            fold_ROC = ROC(result=valid_result, label=valid_label[:, 1])
            print("======================================= AUC:"+str(fold_ROC.auc))
            result_list[index:index+fold_size, 0] = valid_result
            label_list[index:index+fold_size, 0] = valid_label[:, 1]
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

    def save_params_data(self, PATH, matrix):
        fwrite = open(PATH, 'w')
        Shape = matrix.shape

        if len(Shape) == 1:
            tmp_line = str(matrix[0])
            for col in range(1, Shape[0]):
                tmp_line = tmp_line + ' ' + str(matrix[col])
            fwrite.write(tmp_line)
        else:
            for row in range(Shape[0]):
                tmp_line = str(matrix[row][0])
                for col in range(1, Shape[1]):
                    tmp_line = tmp_line + ' ' + str(matrix[row][col])
                if row != Shape[0] - 1:
                    tmp_line = tmp_line + '\n'
                fwrite.write(tmp_line)
        fwrite.close()

    def save_params(self, baseDir):
        for n in range(len(self.n_neurons) - 1):
            w_filename = baseDir + 'layer_' + str(n) + '_weight_' + str(self.n_neurons[n]) + '_' + str(
                self.n_neurons[n + 1]) + '.txt'
            b_filename = baseDir + 'layer_' + str(n) + '_bias_' + str(self.n_neurons[n + 1]) + '.txt'
            # print(sess.run(W['layer_0']))
            w_matrix = self.sess.run(self.weight_list['layer_' + str(n)])
            b_vector = self.sess.run(self.bias_list['layer_' + str(n)])
            self.save_params_data(PATH=w_filename, matrix=w_matrix)
            self.save_params_data(PATH=b_filename, matrix=b_vector)












