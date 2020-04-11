from gbrbm import GBRBM
import BaseOperation as BO
import matplotlib.pyplot as plt


if __name__ == '__main__':
    baseDir = 'F:\\RenLab\\HPV_classification\\train_data\\'
    data_file = baseDir+'train_matrix_pca_200.txt'
    data_matrix = BO.load_np_matrix(data_file, 528, 200)

    HPV_rbm = GBRBM(n_visible=200, n_hidden=150, learning_rate=0.001, momentum=0.95, use_tqdm=True)
    err = HPV_rbm.fit(data_matrix, n_epoches=500, batch_size=10)
    plt.plot(err)
    plt.show()
