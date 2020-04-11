from sklearn.decomposition import PCA
import numpy as np
import BaseOperation as BO
from sklearn.externals import joblib


def pca4matrix(orig_matrix, dim):
    pca = PCA(n_components=dim, whiten=True)
    pca.fit(orig_matrix)
    return pca

# X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#
# Y = pca4matrix(X, 1)
#
# print(Y)

if __name__ == '__main__':
    phase = 'valid'

    if phase == 'train':
        baseDir = 'F:\\RenLab\\HPV_classification\\'
        train_matix_file = baseDir + 'train_data\\train_matrix.txt'
        train_matrix_data = BO.read_matrix_data(train_matix_file, 528)
        feat_dim = 800
        PCA_model = pca4matrix(train_matrix_data, feat_dim)

        model_file = baseDir + 'train_data\\PCA_model_' + str(feat_dim) + '.model'
        joblib.dump(PCA_model, model_file)

        load_model = joblib.load(model_file)
        train_matrix_pca = load_model.transform(train_matrix_data)
        BO.save_pca_matrix(baseDir + 'train_data\\train_matrix_pca_' + str(feat_dim) + '.txt', train_matrix_pca)

        print('PCA matrix size: ' + str(np.shape(train_matrix_pca)))

    if phase == 'valid':
        baseDir = 'F:\\RenLab\\HPV_classification\\'
        train_matix_file = baseDir + 'valid_data\\valid_matrix.txt'
        train_matrix_data = BO.read_matrix_data(train_matix_file, 122)
        feat_dim = 200

        model_file = baseDir + 'valid_data\\PCA_model_' + str(feat_dim) + '.model'

        load_model = joblib.load(model_file)
        train_matrix_pca = load_model.transform(train_matrix_data)
        BO.save_pca_matrix(baseDir + 'valid_data\\valid_matrix_pca_' + str(feat_dim) + '.txt', train_matrix_pca)

        print('PCA matrix size: ' + str(np.shape(train_matrix_pca)))

