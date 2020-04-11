from roc import ROC
import BaseOperation as BO
import numpy as np

if __name__  == '__main__':
    baseDir = 'F:\\RenLab\\zuolaoshi\\HPV-RBM\\HPV-RBM\\RF\\independ\\repeat_1\\'
    label = BO.load_list(baseDir + 'label.txt')
    result = BO.load_list(baseDir + 'result.txt')

    tmp_roc = ROC(result=result, label=label)
    print(tmp_roc.auc)