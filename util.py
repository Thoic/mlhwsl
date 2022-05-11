# create a confusion matrix and determine tpr, fpr
def confuse_matrix(y_predict, y_test):
    conf_mat = [[0,0],
                [0,0]]
    for h, c in zip(y_predict, y_test):
        if h == -1. and c == -1.:
            conf_mat[0][0] += 1
        elif h == 1. and c == -1.:
            conf_mat[0][1] += 1
        elif h == -1. and c == 1.:
            conf_mat[1][0] += 1
        elif h == 1. and c == 1.:
            conf_mat[1][1] += 1
    
    tpr = conf_mat[1][1]/(conf_mat[1][1]+conf_mat[1][0])
    fpr = conf_mat[0][1]/(conf_mat[0][0]+conf_mat[0][1])

    return conf_mat, tpr, fpr