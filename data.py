import re
import BaseOperation as BO

class Data:
    def __init__(self, matrix_file, label_file):
        self.matrix = None
        self.label = {}
        self.label_list = []
        self.datas, self.samples, self.genes = None, None, None
        self.matrix_file = matrix_file
        self.label_file = label_file
        self.get_data()
        self.get_label()
        self.get_label_list()
        self.sample_num = len(self.samples)

    def save_matrix_file(self, PATH, data_dict, gene_list, sample_list):
        fwrite = open(PATH, 'w')
        fwrite.write('sample_name')
        for tmp_gene in gene_list:
            fwrite.write('  ' + tmp_gene)
        fwrite.write('\n')
        for tmp_sample in sample_list:
            fwrite.write(tmp_sample)
            for tmp_gene in gene_list:
                if tmp_gene not in data_dict[tmp_sample].keys():
                    print(tmp_gene)
                tmp_value = data_dict[tmp_sample][tmp_gene]
                fwrite.write('  ' + tmp_value)
            fwrite.write('\n')
        fwrite.close()

    def save_matrix_file_no_taps(self, PATH, data_dict, gene_list, sample_list):
        fwrite = open(PATH, 'w')

        for tmp_sample in sample_list:
            for tmp_gene in gene_list:
                if tmp_gene not in data_dict[tmp_sample].keys():
                    print(tmp_gene)
                tmp_value = data_dict[tmp_sample][tmp_gene]
                fwrite.write(tmp_value + ' ')
            fwrite.write('\n')
        fwrite.close()

    def get_filted_dict(self, gene_list_new):
        dict_new = {}
        for one_sample in self.datas.keys():
            dict_new[one_sample] = {}
            for one_gene in gene_list_new:
                dict_new[one_sample][one_gene] = self.datas[one_sample][one_gene]
        return dict_new

    def get_data(self):
        self.datas, self.samples, self.genes = self.get_fine_matrix(filename=self.matrix_file)
        self.matrix = BO.read_matrix_data(filename=self.matrix_file, sample_num=len(self.samples))

    def get_label(self):
        fid = open(self.label_file)
        line = fid.readline()
        line = fid.readline()
        while line:
            tmp_arr = line.split()
            self.label[tmp_arr[0]] = int(tmp_arr[1])
            line = fid.readline()

    def get_label_list(self):
        for elem in self.samples:
            self.label_list.append(self.label[elem])

    def get_fine_matrix(self, filename):
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
                print('this line is not valid, line:' + str(sample_count))
                break
            tmp_sample = tmp_arr[0]
            data_dict[tmp_sample] = {}
            sample_list.append(tmp_sample)
            for i in range(1, len(tmp_arr)):
                data_dict[tmp_sample][gene_list[i - 1]] = tmp_arr[i]

            sample_count += 1
            line = fid.readline()

        return data_dict, sample_list, gene_list


def filt_label_data(dict_1, sample_1, dict_2, sample_2):
    dict_new = {}
    for tmp_sample_1 in sample_1:
        dict_new[tmp_sample_1] = dict_1[tmp_sample_1]
    for tmp_sample_2 in sample_2:
        dict_new[tmp_sample_2] = dict_2[tmp_sample_2]
    return dict_new


def load_info_list(filename):
    fid = open(filename, 'r')
    List = []
    curline = fid.readline()
    while curline:
        List.append(curline.replace('\n', ''))
        curline = fid.readline()
    fid.close()
    return List


def transOneSetDict(dict1, dict2, gene_list):
    dict_new = {}
    for sample in dict1.keys():
        dict_new[sample] = {}
        for tmpGene in gene_list:
            dict_new[sample][tmpGene] = dict1[sample][tmpGene]
    for sample in dict2.keys():
        dict_new[sample] = {}
        for tmpGene in gene_list:
            dict_new[sample][tmpGene] = dict2[sample][tmpGene]
    return dict_new


def intersect_gene_list(list1, list2):
    list_new = []
    for elem in list1:
        if elem in list2:
            list_new.append(elem)
    return list_new


def get_valid_label(filename):
    fid = open(filename)
    line = fid.readline()
    line = fid.readline()
    sample_list = []
    label_dict = {}
    while line:
        tmp_arr = line.split()
        label_dict[tmp_arr[0]] = int(tmp_arr[1])
        sample_list.append(tmp_arr[0])
        line = fid.readline()
    return sample_list, label_dict


def get_train_label(filename):
    fid = open(filename)
    line = fid.readline()
    line = fid.readline()
    sample_list = []
    label_dict = {}
    while line:
        tmp_arr = line.split()
        label = ''
        if tmp_arr[2] == 'neg':
            label = 0
        else:
            if tmp_arr[2] == 'pos':
                label = 1
        label_dict[tmp_arr[0]] = label
        sample_list.append(tmp_arr[0])
        line = fid.readline()
    return sample_list, label_dict


def get_matrix(filename, flag):
    fid = open(filename)
    line = fid.readline()
    sample_arr = line.split()
    check_num = len(sample_arr)
    sample_list = []
    gene_list = []
    if flag == 'orig':
        for i in range(1, len(sample_arr)):
            sample_list.append(sample_arr[i])
    else:
        if flag == 'TCGA':
            for i in range(1, len(sample_arr)):
                match = re.match(r'(.+)-(.+)-(.+)-(.+)-(.+)-(.+)-(.+)', sample_arr[i])
                tmp_name = match.group(1)
                for j in range(2, 4):
                    tmp_name = tmp_name + '-' + match.group(j)
                sample_list.append(tmp_name)
    print('finished line 1')
    line = fid.readline()
    data_dict = {}
    sample_count = 0
    while line:
        if sample_count % 2000 == 0:
            print('processing line:'+str(sample_count))
        tmp_arr = line.split()
        if len(tmp_arr) != check_num:
            print('this line is not valid, line:'+str(sample_count))
            break
        tmp_gene = tmp_arr[0]
        gene_list.append(tmp_gene)
        for i in range(1, len(tmp_arr)):
            if sample_list[i-1] not in data_dict.keys():
                data_dict[sample_list[i-1]] = {}
            data_dict[sample_list[i - 1]][tmp_gene] = tmp_arr[i]
            # data_dict[sample_list[i - 1]].append(tmp_arr[i])
        sample_count += 1
        line = fid.readline()

    return sample_list, gene_list, data_dict

# ------- training data ---------------------------------------------------
if __name__ == '__main__':
    baseDir = 'F:\\RenLab\\HPV_classification\\'
    tcga_matrix_file = baseDir + 'TCGA_HNSC.tumor_Rsubread_FPKM.txt'
    seiwert_matrix_file = baseDir + 'Seiwert.ProcessedData.gene.v2.134.txt'
    tcga_sample_list, tcga_gene_list, tcga_data_dict = get_matrix(tcga_matrix_file, 'TCGA')
    seiwert_sample_list, seiwert_gene_list, seiwert_data_dict = get_matrix(seiwert_matrix_file, 'orig')

    gene_new_train = intersect_gene_list(tcga_gene_list, seiwert_gene_list)

    # ------- train label --------------------------------------------------------
    tcga_sample_label, tcga_label_dict = get_train_label(baseDir + 'TCGA_group_allocation.txt')
    seiwert_sample_label, seiwert_label_dict = get_train_label(baseDir + 'AGI_group_allocation.new.txt')

    train_sample_tcga = intersect_gene_list(tcga_sample_list, tcga_sample_label)
    train_sample_seiwert = intersect_gene_list(seiwert_sample_list, seiwert_sample_label)
    train_label_dict = filt_label_data(tcga_label_dict, train_sample_tcga, seiwert_label_dict, train_sample_seiwert)

    train_sample_seiwert.extend(train_sample_tcga)
    train_sample = train_sample_seiwert

    # ------- validating data ----------------------------------------------------
    cellline_valid_file = baseDir + 'valid\\Cellline.ProcessedData.gene.nofilter.txt'
    pyeon_valid_file = baseDir + 'valid\\Pyeon_expr_rma_gene.txt'
    cellline_sample_list, cellline_valid_genelist, cellline_valid_datadict = get_matrix(cellline_valid_file, 'orig')
    pyeon_sample_list, pyeon_valid_genelist, pyeon_valid_datadict = get_matrix(pyeon_valid_file, 'orig')

    gene_new_valid = intersect_gene_list(cellline_valid_genelist, pyeon_valid_genelist)

    # ------- validating label ------------------------------------------------------------
    cellline_sample_label, cellline_label_dict = get_valid_label(baseDir + 'valid\\cellline_HPV.txt')
    pyeon_sample_label, pyeon_label_dict = get_valid_label(baseDir + 'valid\\pyeonLabel.txt')

    valid_sample_cellline = intersect_gene_list(cellline_sample_list, cellline_sample_label)
    valid_sample_pyeon = intersect_gene_list(pyeon_sample_list, pyeon_sample_label)
    valid_label_dict = filt_label_data(cellline_label_dict, valid_sample_cellline, pyeon_label_dict, valid_sample_pyeon)

    valid_sample_cellline.extend(valid_sample_pyeon)
    valid_sample = valid_sample_cellline

    # -------- final gene list and data dict ----------------------------------------------
    final_list = intersect_gene_list(gene_new_train, gene_new_valid)

    train_dict = transOneSetDict(tcga_data_dict, seiwert_data_dict, final_list)

    valid_dict = transOneSetDict(cellline_valid_datadict, pyeon_valid_datadict, final_list)

    # -------- save matrix data ----------------------------------------------------------------
    BO.save_matrix_file(baseDir + 'train_data\\train_matrix.txt', train_dict, final_list, train_sample)
    BO.save_label_file(baseDir + 'train_data\\train_label.txt', train_label_dict, train_sample)

    # BO.save_matrix_file(baseDir+'valid_data\\valid_matrix.txt', valid_dict, final_list, valid_sample)
    # BO.save_label_file(baseDir+'valid_data\\valid_label.txt', valid_label_dict, valid_sample)




