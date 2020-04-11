from gbrbm import GBRBM
import numpy as np
import BaseOperation as BO


class HpvRbm:
    def __init__(self, inputs, n_visible, n_hidden, n_epoch, batch_size, layer_count):
        self.gbrbm = GBRBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.001, momentum=0.95, use_tqdm=True)
        self.data = inputs
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.layer_count = layer_count

    def run(self):
        return self.gbrbm.fit(self.data, n_epoches=self.n_epoch, batch_size=self.batch_size)

    def save_params(self, fileDir):
        return self.gbrbm.save_params(filename=fileDir+'rbm_layer_'+str(self.layer_count))

    def compute_hidden_states(self, input_data):
        return self.gbrbm.compute_hidden_states(inputs=input_data)

    def save_epoch_errors(self, PATH):
        BO.writeList2txt(PATH=PATH, list=self.gbrbm.iter_error)

    def save_matrix(self, PATH, matrix):
        [sample_num, feat_num] = np.shape(matrix)
        fwrite = open(PATH, 'w')
        for row in range(sample_num):
            for col in range(feat_num - 1):
                fwrite.write(str(matrix[row][col]) + ' ')
            fwrite.write(str(matrix[row][col]) + '\n')
        fwrite.close()



