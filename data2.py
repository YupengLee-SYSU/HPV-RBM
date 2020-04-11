import numpy as np
import re


def load_info_list(filename):
    fid = open(filename, 'r')
    List = []
    curline = fid.readline()
    while curline:
        List.append(curline.replace('\n', ''))
        curline = fid.readline()
    fid.close()
    return List


def save_info_list(filename, list):
    fid = open(filename, 'w')
    for elem in list:
        # fid.write(str(elem)+'\n')
        fid.write(str(elem))
    fid.close()


def get_matrix_info(filename):
    fid = open(filename)
    line = fid.readline()
    gene_arr = line.split()
    check_num = len(gene_arr)
    sample_list = []
    gene_list = []

    for i in range(1, len(gene_arr)):
        gene_list.append(gene_arr[i])

    print('finished line 1')
    line = fid.readline()
    data_dict = {}
    sample_count = 0
    while line:
        if sample_count % 10 == 0:
            print('processing line:'+str(sample_count))
        tmp_arr = line.split()
        if len(tmp_arr) != check_num:
            print('this line is not valid, line:'+str(sample_count))
            break
        tmp_sample = tmp_arr[0]
        sample_list.append(tmp_sample)

        sample_count += 1
        line = fid.readline()

    return sample_list, gene_list


def get_matrix(filename):
    fid = open(filename)
    line = fid.readline()
    sample_arr = line.split()
    check_num = len(sample_arr)
    sample_list = []
    gene_list = []

    for i in range(1, len(sample_arr)):
        sample_list.append(sample_arr[i])

    print('finished line 1')
    line = fid.readline()
    data_dict = {}
    gene_count = 0
    while line:
        if gene_count % 2000 == 0:
            print('processing line:'+str(gene_count))
        tmp_arr = line.split()
        if len(tmp_arr) != check_num:
            print('this line is not valid, line:'+str(gene_count))
            break
        tmp_gene = tmp_arr[0]
        gene_list.append(tmp_gene)
        for i in range(1, len(tmp_arr)):
            if sample_list[i-1] not in data_dict.keys():
                data_dict[sample_list[i-1]] = {}
            data_dict[sample_list[i - 1]][tmp_gene] = tmp_arr[i]
            # data_dict[sample_list[i - 1]].append(tmp_arr[i])
        gene_count += 1
        line = fid.readline()

    return sample_list, gene_list, data_dict


def get_matrix_from_gene(filename, gene_list_ref):
    fid = open(filename)
    line = fid.readline()
    sample_arr = line.split()
    check_num = len(sample_arr)
    sample_list = []
    gene_list = []

    for i in range(1, len(sample_arr)):
        sample_list.append(sample_arr[i])

    print('finished line 1')
    line = fid.readline()
    data_dict = {}
    gene_count = 0
    while line:
        # if gene_count % 2000 == 0:
        #     print('processing line:'+str(gene_count))
        tmp_arr = line.split()
        if len(tmp_arr) != check_num:
            print('this line is not valid, line:'+str(gene_count))
            break
        tmp_gene = tmp_arr[0]

        if tmp_gene in gene_list_ref:
            gene_list.append(tmp_gene)
            for i in range(1, len(tmp_arr)):
                if sample_list[i - 1] not in data_dict.keys():
                    data_dict[sample_list[i - 1]] = {}
                data_dict[sample_list[i - 1]][tmp_gene] = tmp_arr[i]
                # data_dict[sample_list[i - 1]].append(tmp_arr[i])
            gene_count += 1

        line = fid.readline()

    return data_dict


def get_sample_from_labelFiles(label_file):
    sample_list = []
    fid = open(label_file, 'r')
    curline = fid.readline()
    curline = fid.readline()
    while curline:
        arr = curline.split()
        sample_list.append(arr[0])
        curline = fid.readline()
    return sample_list


def get_fine_matrix(filename):
    fid = open(filename)
    line = fid.readline()
    gene_arr = line.split()
    check_num = len(gene_arr)
    gene_list = []
    sample_list = []

    for i in range(1, len(gene_arr)):
        gene_list.append(gene_arr[i])

    print('finished line 1')
    line = fid.readline()
    data_dict = {}
    sample_count = 0
    while line:
        # if sample_count % 2000 == 0:
        # print('processing line:'+str(sample_count))
        tmp_arr = line.split()
        if len(tmp_arr) != check_num:
            print('this line is not valid, line:'+str(sample_count))
            break
        tmp_sample = tmp_arr[0]
        data_dict[tmp_sample] = {}
        sample_list.append(tmp_sample)
        for i in range(1, len(tmp_arr)):
            data_dict[tmp_sample][gene_list[i-1]] = tmp_arr[i]

        sample_count += 1
        line = fid.readline()

    # return data_dict
    return sample_list


def get_overlap_info(list1, list2):
    count = 0
    list_new = []
    for elem in list2:
        if elem in list1:
            list_new.append(elem)
            count = count + 1
    return count, list_new


def get_sample_overlap_info(list1, list2):
    overlap_num = 0
    list_new = []
    for elem in list2:
        if elem not in list1:
            # print(elem)
            list_new.append(elem)
        else:
            overlap_num += 1
    return list_new, overlap_num


def filtDataFromGeneList(data_dict, gene_list, data_new):
    for oneSample in data_dict.keys():
        data_new[oneSample] = {}
        for tmp_gene in gene_list:
            data_new[oneSample][tmp_gene] = data_dict[oneSample][tmp_gene]
    return data_new


def save_filted_matrix(PATH, data_dict, gene_list):
    fwrite = open(PATH, 'w')
    fwrite.write('sample_name')
    for tmp_gene in gene_list:
        fwrite.write('  '+tmp_gene)
    fwrite.write('\n')

    for tmp_sample in data_dict.keys():
        fwrite.write(tmp_sample)
        for tmp_gene in gene_list:
            tmp_value = data_dict[tmp_sample][tmp_gene]
            fwrite.write('  '+tmp_value)
        fwrite.write('\n')
    fwrite.close()


if __name__ == '__main__':
    baseDir = 'F:\\RenLab\\HPV_classification\\'
    file_list = ['GSE40774_GPL13497', 'GSE55543_GPL17077', 'GSE55544_GPL17077',
                 'GSE55545_GPL17077', 'GSE55546_GPL17077', 'GSE55547_GPL17077', 'GSE55548_GPL17077', 'GSE55549_GPL17077']
    # for filename in file_list:
    #     tmp_file_name = baseDir + filename + '_express.txt'
    #     tmp_sample_list, tmp_gene_list, data_dict = get_matrix(tmp_file_name, 'orig')
    #     save_info_list(baseDir+'sample\\'+filename+'.txt', tmp_sample_list)
    #     save_info_list(baseDir+'gene\\'+filename+'.txt', tmp_gene_list)
    #     print('======================> finish file:'+filename)

    # baseDir = 'F:\\RenLab\\HPV_classification\\'
    # train_file = baseDir+'train_data\\train_matrix.txt'
    # train_sample, train_gene = get_matrix_info(train_file)
    # save_info_list(baseDir+'NEW\\agilient\\orig\\train_sample.txt', train_sample)
    # save_info_list(baseDir+'NEW\\agilient\\orig\\train_gene.txt', train_gene)
    #
    # valid_file = baseDir+'valid_data\\valid_matrix.txt'
    # valid_sample, valid_gene = get_matrix_info(valid_file)
    # save_info_list(baseDir + 'NEW\\agilient\\orig\\valid_sample.txt', valid_sample)
    # save_info_list(baseDir + 'NEW\\agilient\\orig\\valid_gene.txt', valid_gene)

    # train_gene_file = baseDir+'orig\\train_gene.txt'
    # train_gene_list = load_info_list(train_gene_file)
    # valid_gene_file = baseDir+'orig\\valid_gene.txt'
    # valid_gene_list = load_info_list(valid_gene_file)
    #
    # list_new = train_gene_list
    # for filename in file_list:
    #     tmp_gene_file = baseDir+'gene\\'+filename+'.txt'
    #     tmp_list = load_info_list(tmp_gene_file)
    #     overlap_num, list_new = get_overlap_info(list_new, tmp_list)
    #     print('file: '+filename+', tmp length:'+str(len(tmp_list))+', orig gene num:'+str(len(train_gene_list))+', overlap num:'+str(overlap_num))
    # print(len(list_new))
    # save_info_list(baseDir+'final\\gene.txt', list_new)

    # train_sample_file = baseDir+'orig\\train_sample.txt'
    # train_sample_list = load_info_list(train_sample_file)
    # valid_sample_file = baseDir+'orig\\valid_sample.txt'
    # valid_sample_list = load_info_list(valid_sample_file)
    # train_sample_list.extend(valid_sample_list)
    #
    # for fileName in file_list:
    #     tmp_sample_file = baseDir+'sample\\'+fileName+'.txt'
    #     tmp_list = load_info_list(tmp_sample_file)
    #     list_filted, overlap_num = get_sample_overlap_info(train_sample_list, tmp_list)
    #     print('file: '+fileName+', tmp length:'+str(len(tmp_list))+', orig sample num:'+str(len(tmp_sample_file))
    #           + ', overlap num:' + str(overlap_num) + 'final length:'+str(len(list_filted)))

    # for fileName in file_list:
    #     tmp_sample_file = baseDir+'sample\\'+fileName+'.txt'
    #     tmp_list = load_info_list(tmp_sample_file)
    #     train_sample_list.extend(tmp_list)
    #     print('finished file:'+fileName)
    # save_info_list(baseDir+'final\\samples.txt', train_sample_list)

    # baseDir = 'F:\\RenLab\\HPV_classification\\data_test\\'
    # excellist = load_info_list(baseDir+'newLabelFromExcel.txt')
    # sample_list = load_info_list(baseDir+'sampleListFromMatrix.txt')
    #
    # check_list, overlap_num = get_sample_overlap_info(sample_list, excellist)
    # save_info_list(baseDir+'check_list.txt', check_list)
    # print('excel_list length:'+str(len(excellist)))
    # print('matrix_sample length:'+str(len(sample_list)))
    # print(overlap_num)

#     ======================================== valid data =========================================================
#     valid_matrix_old = baseDir+'valid_data\\valid_matrix.txt'
#     valid_matrix = get_fine_matrix(valid_matrix_old)
#     gene_list = load_info_list(baseDir+'DATA2\\valid\\gene.txt')
#
#     matrix_dict = {}
#     valid_matrix_filted = filtDataFromGeneList(valid_matrix, gene_list, matrix_dict)
#
#     save_filted_matrix(baseDir+'DATA2\\valid\\valid_matrix.txt', valid_matrix_filted, gene_list)

    # train_matrix_old = baseDir+'train_data\\train_matrix.txt'
    # train_matrix = get_fine_matrix(train_matrix_old)
    # gene_list = load_info_list(baseDir+'DATA2\\train\\gene.txt')
    # matrix_dict = {}
    # train_matrix_filted = filtDataFromGeneList(train_matrix, gene_list, matrix_dict)
    #
    # for fileName in file_list:
    #     print("file name:"+fileName)
    #     file_name = baseDir+'NEW\\agilient\\'+fileName+'_express.txt'
    #     data_dict = get_matrix_from_gene(file_name, gene_list)
    #     train_matrix_filted = dict(train_matrix_filted, **data_dict)
    #
    # save_filted_matrix(baseDir+'DATA2\\train\\train_matrix.txt', train_matrix_filted, gene_list)

    train_matrix_file = baseDir+'DATA2\\train\\train_matrix.txt'
    train_label_file = baseDir+'DATA2\\train\\train_label.txt'
    sampleFromMatrix = get_fine_matrix(train_matrix_file)
    sampleFromLabel = get_sample_from_labelFiles(train_label_file)

    print('matrix sample num:'+str(len(sampleFromMatrix))+', label sample num:'+str(len(sampleFromLabel)))
    # for elem in sampleFromLabel:
    #     if elem not in sampleFromMatrix:
    #         print(elem)
    new_arr = []
    for elem in sampleFromLabel:
        if elem in new_arr:
            print(elem)
        new_arr.append(elem)
