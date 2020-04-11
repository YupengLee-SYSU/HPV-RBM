from data import Data
import BaseOperation as BO
import data as DT


if __name__ == '__main__':
    baseDir = 'F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\rbm_select\\'
    data_file = baseDir+'all_train_matrix.txt'
    label_file = baseDir+'train_label.txt'

    mapData = Data(matrix_file=data_file, label_file=label_file)
    orig_matrix_dict = mapData.datas
    labels = mapData.label
    sample_list = mapData.samples
    optimal_gene_list = DT.load_info_list(baseDir+'gene.txt')

    pos_samples = []
    neg_samples = []

    for tmp_sample in sample_list:
        if labels[tmp_sample] == 1:
            pos_samples.append(tmp_sample)
        if labels[tmp_sample] == 0:
            neg_samples.append(tmp_sample)

    assert len(pos_samples)+len(neg_samples) == len(labels)

    mapData.save_matrix_file_no_taps(PATH=baseDir+'heat_map\\all\\pos_data.txt', data_dict=orig_matrix_dict,
                             gene_list=optimal_gene_list, sample_list=pos_samples)
    mapData.save_matrix_file_no_taps(PATH=baseDir+'heat_map\\all\\neg_data.txt', data_dict=orig_matrix_dict,
                             gene_list=optimal_gene_list, sample_list=neg_samples)