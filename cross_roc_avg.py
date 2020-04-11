from roc import ROC
import BaseOperation as BO
import numpy as np


def get_FPR_TPR_files(baseDir):
    for i in range(10):
        dir = baseDir+'repeat_'+str(i+1)
        result = BO.load_list(dir+'\\result.txt')
        label = BO.load_list(dir+'\\label.txt')
        tmp_roc = ROC(result=np.array(result), label=np.array(label))
        print('repeat:'+str(i+1))
        print('AUC value:'+str(tmp_roc.auc))
        BO.writeList2txt(dir+'\\FPR.txt', tmp_roc.FPR_sorted)
        BO.writeList2txt(dir+'\\TPR.txt', tmp_roc.TPR_sorted)


def get_avg_curve(baseDir):
    curve_set = {}
    for i in range(10):
        dir = baseDir + 'repeat_' + str(i + 1)
        tmp_FPR = BO.load_list(dir+'\\FPR.txt')
        tmp_TPR = BO.load_list(dir+'\\TPR.txt')
        curve_set[str(i+1)] = {}
        curve_set[str(i+1)]['FPR'], curve_set[str(i+1)]['TPR'] = tmp_FPR, tmp_TPR
    FPR_avg = []
    TPR_avg = []
    count = 1000
    gap = 1/count
    for j in range(count+1):
        FPR_avg.append(gap*j)
        TPR_avg.append(avg_tpr(curve_set=curve_set, fpr=gap*j))
    print('AUC for average curve:'+str(computeArea(FPR_avg, TPR_avg)))
    BO.writeList2txt(baseDir+'FPR.txt', FPR_avg)
    BO.writeList2txt(baseDir+'TPR.txt', TPR_avg)


def computeArea(FPR, TPR):
    Area = 0
    p_start, p_stop = 0, 0
    d_up, d_down = 0, 0
    for i in range(len(FPR)):
        p_stop = FPR[i]
        d_down = TPR[i]
        Area += (d_up + d_down) * (p_stop - p_start) * 0.5
        p_start = p_stop
        d_up = d_down
    return Area


def avg_tpr(curve_set, fpr):
    tpr_sum = 0
    # for i in [0,1,2,5,6,7,9]:
    for i in range(10):
        tmp_curve = curve_set[str(i+1)]
        tmp_FPR, tmp_TPR = padding(tmp_curve['FPR']), padding(tmp_curve['TPR'])
        count = 1
        while fpr > tmp_FPR[count]:
            count += 1
        ind1 = count-1
        ind2 = count
        x1, y1 = tmp_FPR[ind1], tmp_TPR[ind1]
        x2, y2 = tmp_FPR[ind2], tmp_TPR[ind2]
        if fpr == x1:
            tmp_tpr = y1
        else:
            tmp_tpr = ((y2 - y1) / (x2 - x1)) * (fpr - x1) + y1
        tpr_sum += tmp_tpr
    return tpr_sum/10


def padding(list0):
    list1 = [0]
    list1.extend(list0)
    list1.append(1)
    return list1


if __name__ == '__main__':
    baseDir = 'F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\'
    # model_type = 'RBM'
    model_type = 'SVM'
    # model_type = 'MLP'
    # model_type = 'RF'
    fold_type = 4
    # mode = 'nfold'
    mode = 'valid'

    tmpdir = ''
    if mode == 'valid':
        tmpdir = baseDir+model_type+'\\independ\\'
    if mode == 'nfold':
        tmpdir = baseDir+model_type+'\\'+str(fold_type)+'_fold\\'

    get_FPR_TPR_files(tmpdir)

    get_avg_curve(tmpdir)


