import numpy as np
from sklearn.feature_selection import VarianceThreshold


# X = np.random.random([10, 4])
# Y = [0,0,1,1,0,0,0,0,0,1]
# names = ['a', 'b', 'c', 'd']
# lr = LinearRegression()
# rfe = RFE(lr, n_features_to_select=1)
# rfe.fit(X, Y)
#
# print_s = rfe.ranking_
#
# print(print_s)

class variance_selection:
    def __init__(self, inputs, names, selected_num):
        self.inputs = inputs
        self.names = np.array(names)
        self.select_num = selected_num
        self.names_selected = np.zeros(selected_num)
        self.varias = []
        self.variModel = VarianceThreshold()
        self.compute()
        self.get_selected_names()

    def compute(self):
        self.variModel.fit(self.inputs)
        self.varias = self.variModel.variances_

    def get_selected_names(self):
        index_sorted = np.argsort(-np.array(self.varias))
        self.names_selected = self.names[index_sorted][0: self.select_num]

    def transform_dicts(self, extra_dict, sampls):
        dict_new = {}
        for one_sample in sampls:
            dict_new[one_sample] = {}
            for elem in self.names_selected:
                dict_new[one_sample][elem] = extra_dict[one_sample][elem]
        return dict_new

# if __name__ == '__main__':
    # train_data =
    # pre_selection_model = RFE_selection()
