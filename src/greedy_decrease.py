from liblinearutil import *
import numpy as np
from random import *
import copy
import subprocess
import time
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter


# ------------------------------------

train_file_name = sys.argv[1]
test_file_name = train_file_name + '.t'
pure_data_name = (train_file_name.split('/'))[len(train_file_name.split('/'))-1]

liblinear_incdec_train_path = '../../liblinear-incdec-2.01/train'
# train_size_rate = 0.25
data_num_size = 10

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

def greedy_Ein (data_name, data_num_size, \
                base_size, base_acc_in, base_acc_out, \
                train_x, train_y, test_x, test_y):
    train_x_copy = copy.deepcopy(train_x)
    train_y_copy = copy.deepcopy(train_y)

    E_in = [1.0]

    '''
    for i in range (data_num_size):
        for j in range (base_size):
            temp_int = randint (0,len(train_x_copy)-1)
            # temp_int = len(train_x_copy)-1 ;
            train_x_copy.pop(temp_int)
            train_y_copy.pop(temp_int)
        write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train')
        cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train'
        subprocess.call (cmd.split())
        m2 = load_model (data_name+'-temp2.train.model')
        # m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;

        p_label , p_acc , p_val = predict(train_y , train_x , m2)
        E_in.append(p_acc[0]/base_acc_in)
        p_label , p_acc , p_val = predict(test_y , test_x , m2)
        E_out.append(p_acc[0]/base_acc_out)
     '''
# ------------------------------------

train_y, train_x = svm_read_problem(train_file_name)
test_y, test_x = svm_read_problem(test_file_name)

train_x, train_y = check_zero(train_x, train_y)
test_x, test_y = check_zero(test_x, test_y)

base_size = int(len(train_y)*0.1/data_num_size) + 1
# base_size = 1 ;

write_svm_file (train_x, train_y, pure_data_name+'-temp.train')

# ------------------------------------
cmd = liblinear_incdec_train_path + ' -s 2 -q ' + pure_data_name + '-temp.train'
subprocess.call (cmd.split())
m = load_model (pure_data_name+'-temp.train.model')

p_label_in, p_acc_in, p_val_in = predict(train_y, train_x, m)
base_acc_in = p_acc_in[0]
p_label_out, p_acc_out, p_val_out = predict(test_y, test_x, m)
base_acc_out = p_acc_out[0]
E_in_0 = [1.0 for i in range(data_num_size+1)]
E_out_0 = [1.0 for i in range(data_num_size+1)]
# ------------------------------------
