from liblinearutil import *
import numpy as np
from random import *
import copy
import subprocess
import time
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter


# ------------------------------------

train_file_name = sys.argv[1]
pickle_path = sys.argv[2]
test_file_name = train_file_name + '.t'
pure_data_name = (train_file_name.split('/'))[len(train_file_name.split('/'))-1]

liblinear_incdec_train_path = '../../liblinear-incdec-2.01/train'
# train_size_rate = 0.25

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

data_num_size = int(len(train_y)*0.1)
#base_size = int(len(train_y)*0.1/data_num_size) + 1
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

remove_order=[]
E_in_1 = []
E_out_1 = []
train_x_copy = copy.deepcopy(train_x)
train_y_copy = copy.deepcopy(train_y)
for it in range(data_num_size+1):
    acc_rank = []
    write_svm_file (train_x_copy, train_y_copy, pure_data_name+'-temp2.train')
    cmd = liblinear_incdec_train_path + ' -s 2 -q ' + pure_data_name + '-temp2.train'
    subprocess.call (cmd.split())

    for i in range(len(train_y_copy)):
        cmd = 'cp ' + pure_data_name + '-temp2.train.model'+ ' ' + pure_data_name + '-temp3.train.model'
        subprocess.call (cmd.split())

        temp_train_x = copy.deepcopy(train_x_copy)
        temp_train_x.pop(i)
        temp_train_y = copy.deepcopy(train_y_copy)
        temp_train_y.pop(i)
        write_svm_file (temp_train_x, temp_train_y, pure_data_name+'-'+'temp3.train')
        cmd = liblinear_incdec_train_path \
                + ' -s 2 -q -i ' \
                + pure_data_name+'-temp3.train.model ' \
                + pure_data_name+'-temp3.train'
        subprocess.call (cmd.split())
        m2 = load_model (pure_data_name+'-'+'temp3.train.model')

        p_label_in, p_acc_in, p_val_in      = predict(temp_train_y, temp_train_x, m2)
        p_label_out, p_acc_out, p_val_out   = predict(test_y, test_x, m2)
        acc_rank.append((p_acc_out[0], p_acc_in[0], i))

        cmd = 'rm ' + pure_data_name + '-temp3.train.model'
        subprocess.call (cmd.split())

    (max_acc_out, max_acc_in, remove_num) = sorted(acc_rank)[-1] #sort by max_acc_out
    remove_order.append(remove_num)
    E_in_1.append(max_acc_in)
    E_out_1.append(max_acc_out)
    train_x_copy.pop(remove_num)
    train_y_copy.pop(remove_num)

E_in_1 = [x / base_acc_in for x in E_in_1]
E_out_1 = [x / base_acc_out for x in E_in_1]


with open(pickle_path) as store:
    pickle.dump((pure_data_name, remove_order, E_out_1, E_in_1), store)
# ------------------------------------
step = round(0.1 / len(E_out_1), 5)
terminal = 1.00 - step * len(E_out_1)
query_num = [1.00 - step * x for x in range(len(E_out_1))]

plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4)
ax = plt.gca()
ax.xaxis.set_major_locator( MultipleLocator(0.01) )

plt.xlabel('% of Data')
plt.ylabel('Acc rate')
plt.xlim(1.00, terminal)
plt.ylim(0.0 , 0.02)
plt.grid()


plt.title(pure_data_name + '_out_' + ('%.3f' % base_acc_out))
plt.plot(query_num, E_out_0, 'k', label='total')
#plt.plot(query_num, E_out_02, 'r', label='noise')
plt.plot(query_num, E_out_1, 'bo--', label='greedy')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3)
plt.savefig(pure_data_name + '_out.png')

plt.cla()

# ----------------------------------------------------------
step = round(0.1 / len(E_in_1), 5)
terminal = 1.00 - step * len(E_in_1)
query_num = [1.00 - step * x for x in range(len(E_in_1))]

plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4)
ax = plt.gca()
ax.xaxis.set_major_locator( MultipleLocator(0.01) )

plt.xlabel('% of Data')
plt.ylabel('Acc rate')
plt.xlim(1.00, terminal)
# plt.ylim(0.8 , 1.2)
plt.grid()


plt.title(pure_data_name + '_in_' + ('%.3f' % base_acc_in))
plt.plot(query_num, E_in_0, 'k', label='total')
#plt.plot(query_num, E_in_02, 'r', label='noise')
plt.plot(query_num, E_in_1, 'bo--', label='greedy')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3)
plt.savefig(pure_data_name + '_in.png')

plt.cla()
