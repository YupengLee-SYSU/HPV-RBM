import numpy as np
import BaseOperation as BO
from feature_selection import variance_selection
from data import Data
import data as DT
from randomForest import RandomForest
from hpvRBM import HpvRbm
from MLP import YpMLP
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from rbm_select import RBM_select


isLinux = False
# phase = 'feature'
phase = 'model'

variance_n = 2500
randomforest_n = 800

if phase == 'feature':
    # =========================================== train data ===================================
    baseDir = 'F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\DATA2\\'
    # baseDir = '/home/liyupeng/HPV-RBM/HPV_data/DATA2/'
    gene_file = DT.load_info_list(baseDir + BO.windows2linuxDirs('train\\gene.txt', isLinux))
    train_label_file = baseDir + BO.windows2linuxDirs('train\\train_label.txt', isLinux)
    train_matrix_file = baseDir + BO.windows2linuxDirs('train\\train_matrix.txt', isLinux)

    train_data = Data(matrix_file=train_matrix_file, label_file=train_label_file)
    train_sample_list = train_data.samples
    train_label_list = train_data.label_list

    varia_selection = variance_selection(inputs=train_data.matrix, names=train_data.genes, selected_num=variance_n)
    gene_sel_1 = varia_selection.names_selected
    print('gene number after variance selection:' + str(len(gene_sel_1)))
    data_dict_sel1 = varia_selection.transform_dicts(extra_dict=train_data.datas, sampls=train_sample_list)

    randomForestModel = RandomForest(BO.dataDict2Matrix(data_dict_sel1, train_sample_list, gene_sel_1),
                                     train_label_list,
                                     gene_sel_1, randomforest_n)
    randomForestModel.run()
    gene_sel_2 = randomForestModel.get_n_selected_features()
    print('gene number after random forest:' + str(len(gene_sel_2)))

    BO.writeList2txt(baseDir + BO.windows2linuxDirs('train\\genes_selected.txt', isLinux), gene_sel_2)

    train_data.save_matrix_file(PATH=baseDir+BO.windows2linuxDirs('train\\train_matrix_'+str(randomforest_n)+'.txt', isLinux), data_dict=data_dict_sel1,
                                gene_list=gene_sel_2, sample_list=train_sample_list)

    # ================================= valid data ==========================================
    valid_label_file = baseDir+BO.windows2linuxDirs('valid\\valid_label.txt', isLinux)
    valid_matrix_file = baseDir+BO.windows2linuxDirs('valid\\valid_matrix.txt', isLinux)
    valid_data = Data(matrix_file=valid_matrix_file, label_file=valid_label_file)
    valid_sample_list = valid_data.samples
    valid_label_list = valid_data.label_list
    print('sample number:'+str(len(valid_sample_list))+', label number:'+str(len(valid_label_list)))

    valid_data.save_matrix_file(PATH=baseDir+BO.windows2linuxDirs('valid\\valid_matrix_'+str(randomforest_n)+'.txt', isLinux), data_dict=valid_data.datas,
                                gene_list=gene_sel_2, sample_list=valid_sample_list)

if phase == 'model':
    #  800 ---> 400 ---> 100 ---> 2
    net_size = [randomforest_n, 600, 300, 2]

    # baseDir = '/home/liyupeng/HPV-RBM/'
    baseDir = 'F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\'

    dataDir = baseDir+BO.windows2linuxDirs('DATA2\\train\\', isLinux)
    train_matrix_file = dataDir+'train_matrix_'+str(randomforest_n)+'.txt'
    train_label_file = dataDir+'train_label.txt'
    train_data = Data(matrix_file=train_matrix_file, label_file=train_label_file)

    train_matrix_data = train_data.matrix

    validDir = baseDir+BO.windows2linuxDirs('DATA2\\valid\\', isLinux)
    valid_matrix_file = validDir+'valid_matrix_'+str(randomforest_n)+'.txt'
    valid_label_file = validDir+'valid_label.txt'
    valid_data = Data(matrix_file=valid_matrix_file, label_file=valid_label_file)

    # train_stage = 'rbm_1'
    # train_stage = 'rbm_2'
    train_stage = 'mlp'
    # train_stage = 'pure_mlp'
    # train_stage = 'gene_select'
    # rbm_1, rbm_2, mlp, pure_mlp
    if train_stage == 'rbm_1':
        Scalor = StandardScaler().fit(train_data.matrix)
        joblib.dump(Scalor, baseDir+BO.windows2linuxDirs('norm_model\\StanderScalor.model', isLinux))
        input_matrix = Scalor.transform(train_data.matrix)

        rbm_model_1 = HpvRbm(inputs=input_matrix, n_visible=randomforest_n, n_hidden=net_size[1]
                             , n_epoch=300, batch_size=20, layer_count=1)
        rbm_model_1.run()

        rbm_model_1.save_epoch_errors(PATH=baseDir+'model\\layer_1\\epoch_error.txt')

        # rbm_model_1.save_params(fileDir=baseDir+BO.windows2linuxDirs('model\\layer_1\\', isLinux))
        # rbm_states = rbm_model_1.compute_hidden_states(input_data=input_matrix)
        # rbm_model_1.save_matrix(dataDir+'matrix_rbm_layer_1.txt', rbm_states)

    if train_stage == 'rbm_2':
        rbm_model_2 = HpvRbm(inputs=BO.load_np_matrix(PATH=dataDir+'matrix_rbm_layer_1.txt', sample_num=train_data.sample_num, feat_dim=net_size[1])
                             , n_visible=net_size[1], n_hidden=net_size[2], n_epoch=300, batch_size=20, layer_count=2)
        rbm_model_2.run()

        rbm_model_2.save_epoch_errors(PATH=baseDir+'model\\layer_2\\epoch_error.txt')

        # rbm_model_2.save_params(fileDir=baseDir+BO.windows2linuxDirs('model\\layer_2\\', isLinux))

    if train_stage == 'mlp':
        params_layer_1 = BO.load_nets_params(modelname=baseDir+BO.windows2linuxDirs('model\\layer_1\\rbm_layer_1', isLinux),
                                             in_size=net_size[0], out_size=net_size[1])
        print('finished params layer_1')
        params_layer_2 = BO.load_nets_params(modelname=baseDir+BO.windows2linuxDirs('model\\layer_2\\rbm_layer_2', isLinux),
                                             in_size=net_size[1], out_size=net_size[2])
        print('finished params layer_2')
        params_all = BO.combine_dicts([params_layer_1, params_layer_2])

        trained_scalar_normer = joblib.load(baseDir+BO.windows2linuxDirs('norm_model\\StanderScalor.model', isLinux))
        valid_matrix_data = trained_scalar_normer.transform(valid_data.matrix)
        train_matrix_data = trained_scalar_normer.transform(train_data.matrix)

        for re_N in [1]:
            HRM_model = YpMLP(train_data=train_matrix_data, train_label=train_data.label_list, params=params_all,
                              valid_data=valid_matrix_data, valid_label=valid_data.label_list, n_neurons=net_size,
                              isSaveModelParams=True)
            HRM_model.run_trainer(Epoch_Num=200, Batch_Size=20)
            BO.writeList2txt(PATH='F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\RBM\\independ\\train_err.txt',
                             list=HRM_model.epoch_train_err)
            BO.writeList2txt(PATH='F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\RBM\\independ\\valid_err.txt',
                             list=HRM_model.epoch_valid_err)

            HRM_model.save_params(baseDir='F:\\RenLab\\zuolaoshi\\deepHPV_web\\use_case\\model\\DBN\\')
            HRM_model.save_valid_results(saveBase=baseDir + 'RBM\\', repeat_n=re_N)

        # for fold_n in [4, 6, 8, 10]:
        #     for repeat in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        #         HRM_model = YpMLP(train_data=train_matrix_data, train_label=train_data.label_list, params=params_all,
        #                           valid_data=valid_matrix_data, valid_label=valid_data.label_list)
        #
        #         HRM_model.cross_val(n_fold=fold_n, repeat_num=repeat, saveBase=baseDir + 'RBM\\')

    if train_stage == 'pure_mlp':
        trained_scalar_normer = joblib.load(baseDir + BO.windows2linuxDirs('norm_model\\StanderScalor.model', isLinux))
        valid_matrix_data = trained_scalar_normer.transform(valid_data.matrix)
        train_matrix_data = trained_scalar_normer.transform(train_data.matrix)

        fold_n = 10
        for repeat in range(1):
            HRM_model = YpMLP(train_data=train_matrix_data, train_label=train_data.label_list, n_layers=3,
                              n_neurons=net_size, valid_data=valid_matrix_data, valid_label=valid_data.label_list)
            HRM_model.run_trainer(Epoch_Num=200, Batch_Size=20)
            # HRM_model.save_valid_results(saveBase=baseDir + 'MLP\\', repeat_n=repeat + 1)

            BO.writeList2txt(PATH='F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\MLP\\independ\\train_err.txt',
                             list=HRM_model.epoch_train_err)
            BO.writeList2txt(PATH='F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\MLP\\independ\\valid_err.txt',
                             list=HRM_model.epoch_valid_err)

            # HRM_model.cross_val(n_fold=fold_n, repeat_num=repeat + 1, saveBase=baseDir + 'MLP\\')

    if train_stage == 'gene_select':
        params_layer_1 = BO.load_nets_params(
            modelname=baseDir + BO.windows2linuxDirs('model\\layer_1\\rbm_layer_1', isLinux),
            in_size=net_size[0], out_size=net_size[1])
        print('finished params layer_1')
        params_layer_2 = BO.load_nets_params(
            modelname=baseDir + BO.windows2linuxDirs('model\\layer_2\\rbm_layer_2', isLinux),
            in_size=net_size[1], out_size=net_size[2])
        print('finished params layer_2')
        params_all = BO.combine_dicts([params_layer_1, params_layer_2])

        trained_scalar_normer = joblib.load(baseDir + BO.windows2linuxDirs('norm_model\\StanderScalor.model', isLinux))
        train_matrix_data = trained_scalar_normer.transform(train_data.matrix)

        re_num = 20

        saveBase = baseDir+'rbm_select\\'
        Sel_module = RBM_select(data=train_matrix_data, label=train_data.label_list, gene_list=train_data.genes,
                                repeat_n=re_num, params=params_all, saveBase=saveBase)
        # Sel_module.run()
        # for re in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # for re in range(20):
        #     Sel_module.get_sorted_gene(sel='auc', repeat_num=re+1)

        for re in range(20):
            Sel_module.gen_sorted_list(repeat_num=re+1)
