# python train.py

from liblinearutil import * ;
import numpy as np ;
from random import * ;
import copy ;
import subprocess ;
import time ;

import matplotlib ;
matplotlib.use('Agg') ;
import matplotlib.pyplot as plt ;
from matplotlib.ticker import MultipleLocator, FuncFormatter ;

# ------------------------------------

def write_svm_file (x , y , file_name) :
	output_file = open (file_name , 'w') ;
	for i in range (len(y)) :
		output_file.write (str(y[i])) ;
		for j in x[i].keys() :
			output_file.write (' ' + str(j) + ':' + str(x[i][j])) ;
		output_file.write ('\n') ;
	return ;

def run (data_name , select_method , data_num_size , base_size , base_error , train_x , train_y , test_x , test_y) :
	train_x_copy = copy.deepcopy(train_x) ;
	train_y_copy = copy.deepcopy(train_y) ;

	E_out = [1.0] ;
	write_svm_file (train_x_copy , train_y_copy , data_name+'-'+'temp2.train') ;
	cmd = '../../liblinear-incdec-2.01/train -s 2 -q ' + data_name+'-'+ 'temp2.train' ;
	subprocess.call (cmd.split()) ;
	m = load_model (data_name+'-'+'temp2.train.model') ;
	# m = train (train_y_copy , train_x_copy , '-s 2 -q') ;
	label , acc , val = predict(train_y_copy , train_x_copy , m) ;

	new_val = list(map(max , val)) ;
	new_val = list(map(abs , new_val)) ;
	train_x_copy = [x for (v , x) in sorted(zip(new_val , train_x_copy)) ] ;
	train_y_copy = [y for (v , y) in sorted(zip(new_val , train_y_copy)) ] ;
	label = [l for (v , l) in sorted(zip(new_val , label)) ] ;
	if (select_method == 1) :
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = randint (0,len(train_x_copy)-1) ;
				train_x_copy.pop(temp_int) ;
				train_y_copy.pop(temp_int) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-'+'temp2.train') ;
			cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-'+ 'temp2.train.model ' + data_name+'-'+ 'temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-'+'temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;

			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_error) ;

	return E_out ;

# ------------------------------------

data_name = 'data3' ;
train_size_rate = 0.25 ;
data_num_size = 10 ;

# ------------------------------------

y , x = svm_read_problem('../../data/' + data_name + '.txt') ;

train_bordor = int(len(x)*train_size_rate) ;
train_x = x[:train_bordor] ;
train_y = y[:train_bordor] ;
test_x = x[train_bordor:] ;
test_y = y[train_bordor:] ;

base_size = int(len(train_x)*0.1/data_num_size) + 1 ;
# base_size = 1 ;

write_svm_file (train_x , train_y , data_name+'-'+'temp.train') ;

# ------------------------------------

cmd = '../../liblinear-incdec-2.01/train -s 2 -q ' + data_name+'-'+ 'temp.train' ;
subprocess.call (cmd.split()) ;
m = load_model (data_name+'-'+'temp.train.model') ;
# m = train(train_y , train_x , '-s 2 -q') ;

p_label , p_acc , p_val = predict(test_y , test_x , m) ;
base_error = p_acc[0] ;
E_out_0 = [1.0 for i in range(data_num_size+1)] ;
# print (test_y[0] , p_label[0] , p_val[0] , len(p_val)) ;

# ------------------------------------

# random
select_method = 1 ;
E_out_1 = run (data_name , select_method , data_num_size , base_size , base_error , train_x , train_y , test_x , test_y) ;
print ('random') ;
print (E_out_1) ;

# ------------------------------------

cmd = 'rm ' + data_name+'-'+ 'temp.train' ;
subprocess.call (cmd.split()) ;
cmd = 'rm ' + data_name+'-'+ 'temp.train.model' ;
subprocess.call (cmd.split()) ;

cmd = 'rm ' + data_name+'-'+ 'temp2.train' ;
subprocess.call (cmd.split()) ;
cmd = 'rm ' + data_name+'-'+ 'temp2.train.model' ;
subprocess.call (cmd.split()) ;

# ------------------------------------

query_num = np.arange(1.00 , 0.89 , -0.01) ;
plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4) ;
ax = plt.gca() ;
ax.xaxis.set_major_locator( MultipleLocator(0.01) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
# ax.yaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('Error rate') ;
plt.title(data_name + '_' + str(base_error)) ;
plt.xlim(0.90 , 1.00) ;
# plt.ylim(0.5 , 1.5) ;
plt.grid () ;

plt.plot(query_num, E_out_0, 'k--', label='total') ;
plt.plot(query_num, E_out_1, 'bo--', label='random') ;
plt.plot(query_num, E_out_2, 'rv--', label='wrong hyper high') ;
plt.plot(query_num, E_out_3, 'g^--', label='correct hyper low') ;
plt.plot(query_num, E_out_4, 'c*--', label='hyper high') ;
plt.plot(query_num, E_out_5, 'mx--', label='hyper low') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(data_name + '_liblinear' + '.png') ;

# ------------------------------------
