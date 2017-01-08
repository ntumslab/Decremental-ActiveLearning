from liblinearutil import *
from random import *
import copy
import subprocess
import time
import sys
import os
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

train_file_name = sys.argv[1]
output_Folder = '../../output/'
pure_data_name = (train_file_name.split('/'))[len(train_file_name.split('/'))-1].replace('.s1', '')
test_file_name = '../../data/' + pure_data_name + '.t'
tmp_file_path = '/tmp/cwtsai/Dec-Ative/'

liblinear_incdec_train_path = '~/liblinear-incdec-2.01/train'

# ------------------------------------
def check_zero (x, y):
    for _i in range (len(y)):
        if not x[_i]:
            x[_i][1] = 0
    return (x, y)

def write_svm_file (x, y, file_name):
    output_file = open(file_name, 'w')
    for _i in range (len(y)):
        output_file.write (str(y[_i]))
        temp_list = sorted(list(x[_i].keys()) )
        for _j in temp_list:
            output_file.write (' ' + str(_j) + ':' + str(x[_i][_j]))
        output_file.write ('\n')

def partGreedy(X, Y, X_valid, Y_valid):
    N = len(Y)
    remain_dict  = range(N)
    remove_order=[]
    X_copy = copy.deepcopy(X)
    Y_copy = copy.deepcopy(Y)

    for it in range(N):
        acc_rank = []
        write_svm_file (X_copy, Y_copy, tmp_file_path + pure_data_name+'-temp2-p.train')
        cmd = liblinear_incdec_train_path + ' -s 2  -q ' +  tmp_file_path + pure_data_name + '-temp2-p.train'
        #subprocess.check_call (cmd.split())
        os.system(cmd)

        for i in range(len(Y_copy)):
            cmd = 'cp ' \
                  +  pure_data_name + '-temp2-p.train.model'+ ' ' \
                  +  pure_data_name + '-temp3-p.train.model'
            #subprocess.check_call (cmd.split())
            os.system(cmd)

            X_copy_tmp = copy.deepcopy(X_copy)
            X_copy_tmp.pop(i)
            Y_copy_tmp = copy.deepcopy(Y_copy)
            Y_copy_tmp.pop(i)

            write_svm_file (X_copy_tmp, Y_copy_tmp,  tmp_file_path + pure_data_name+'-'+'temp3-p.train')
            cmd = liblinear_incdec_train_path \
                    + ' -s 2 -q -i ' \
                    +  pure_data_name+'-temp3-p.train.model' + ' ' \
                    + tmp_file_path+pure_data_name+'-temp3-p.train'
            #subprocess.check_call (cmd.split())
            os.system(cmd)
            m2 = load_model (pure_data_name+'-'+'temp3-p.train.model')

            p_label_in, p_acc_in, p_val_in      = predict(Y_copy_tmp, X_copy_tmp, m2)
            p_label_out, p_acc_out, p_val_out   = predict(Y_valid, X_valid, m2)
            acc_rank.append((p_acc_out[0], p_acc_in[0], i))

            cmd = 'rm ' + pure_data_name + '-temp3-p.train.model'
            #subprocess.check_call (cmd.split())
            os.system(cmd)

        cmd = 'rm ' + tmp_file_path + pure_data_name + '-temp2-p.train.model'
        #subprocess.check_call (cmd.split())
        os.system(cmd)
        cmd = 'rm ' + pure_data_name + '-temp2-p.train'
        #subprocess.check_call (cmd.split())
        os.system(cmd)

        (max_acc_out, max_acc_in, remove_num) = sorted(acc_rank, key=lambda x:x[0])[-1] #sort by p_acc_out
        remove_order.append(remain_dict.pop(remove_num)) #since the real remove index is in the remain_dict
        #E_in_1.append(max_acc_in)
        #E_out_1.append(max_acc_out)
        X_copy.pop(remove_num)
        Y_copy.pop(remove_num)

    remove_order_list = np.zeros(N)
    for i, v in enumerate(remove_order):
        remove_order_list[v] = i
    return remove_order_list
# ------------------------------------
if __name__ == '__main__':
    train_y, train_x = svm_read_problem(train_file_name)
    train_x, train_y = check_zero(train_x, train_y)
    N = len(train_y)
    folds_len = np.zeros(5).astype('int')

    print('Divid to 5 folds by % 5 ...')
    folds_x = [ [] for _ in range(5)]
    folds_y = [ [] for _ in range(5)]
    for i in range(N):
        for c in range(5):
            if i % 5 == c:
                folds_x[c].append(train_x[i]);  folds_y[c].append(train_y[i])
                break
    for i in range(5):
        folds_len[i] = len(folds_y[i])

    print('Cross validation ...')
    final_order_list = [[] for _ in range(N)]

    for iteration  in range(1):
        print('Iter {} ...' .format(iteration))
        process_num = range(5)
        process_num.pop(iteration)

        tmp_x, tmp_y, mapping = [], [], []
        for c in process_num:
            tmp_x = tmp_x + folds_x[c]
            tmp_y = tmp_y + folds_y[c]
            mapping = mapping + [ c + 5*i for i in range(folds_len[c]) ]
        remove_order_list = partGreedy(tmp_x, tmp_y, folds_x[iteration], folds_y[iteration])
        del tmp_x; del tmp_y

        for i in range(remove_order_list):
            final_order_list[ mapping[i] ].append(remove_order_list[i])
        pickle.dump(final_order_list, open('final_order_{}.p'.format(pure_data_name) ,'wb'), protocol=2)

    '''
    tmp_x = folds_x[0] + folds_x[1] + folds_x[2] + folds_x[3]
    tmp_y = folds_y[0] + folds_y[1] + folds_y[2] + folds_y[3]
    mapping = [ 0 + 5*i for i in range(folds_len[0]) ] +\
          [ 1 + 5*i for i in range(folds_len[1]) ] +\
          [ 2 + 5*i for i in range(folds_len[2]) ] +\
          [ 3 + 5*i for i in range(folds_len[3]) ]
    '''
