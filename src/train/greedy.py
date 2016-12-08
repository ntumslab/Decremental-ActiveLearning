# python train.py train_file_name

from liblinearutil import * ;
import numpy as np ;
from random import * ;
import copy ;
import subprocess ;
import time ;
import sys ;
from scipy.spatial import distance ;

import matplotlib ;
matplotlib.use('Agg') ;
import matplotlib.pyplot as plt ;
from matplotlib.ticker import MultipleLocator, FuncFormatter ;

# ------------------------------------

def check_zero (x , y) :
	for i in range (len(y)) :
		if not x[i] :
			x[i][1] = 0 ;
	return x , y ;

def write_svm_file (x , y , file_name) :
	output_file = open (file_name , 'w') ;
	for i in range (len(y)) :
		output_file.write (str(y[i])) ;
		temp_list = sorted(list(x[i].keys()) ) ;
		for j in temp_list :
			output_file.write (' ' + str(j) + ':' + str(x[i][j])) ;
		output_file.write ('\n') ;
	return ;

def run (data_name , select_method , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) :
	train_x_copy = copy.deepcopy(train_x) ;
	train_y_copy = copy.deepcopy(train_y) ;
	
	write_svm_file (train_x_copy , train_y_copy , data_name+'-temp.train') ;
	cmd = '../package/liblinear-incdec-2.01/train -s 2 -q ' + data_name+'-temp.train' ;
	subprocess.call (cmd.split()) ;
	
	A_in = [1.0] ;
	A_out = [1.0] ;
	pop_list = [] ;
	
	train_data_len = len(train_y) ;
	# train_data_len = 4 ;
	if (select_method == 1) :
		train_order = [i for i in range(len(train_x_copy))] ;
		for i in range (train_data_len-1) :
			print (i) ;
			temp_max = [0 , 0.0] ;
			for j in range (len(train_x_copy)) :
				train_x_copy2 = copy.deepcopy(train_x_copy) ;
				train_y_copy2 = copy.deepcopy(train_y_copy) ;
				
				train_x_copy2.pop(j) ;
				train_y_copy2.pop(j) ;
				
				write_svm_file (train_x_copy2 , train_y_copy2 , data_name+'-temp2.train') ;
				cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp.train.model ' + data_name+'-temp2.train' ;
				subprocess.call (cmd.split()) ;
				m2 = load_model (data_name+'-temp2.train.model') ;

				p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
				if (p_acc[0] > temp_max[1]) :
					temp_max[1] = p_acc[0] ;
					temp_max[0] = j ;
			
			pop_list.append(train_order.pop(temp_max[0])) ;
			train_x_copy.pop(temp_max[0]) ;
			train_y_copy.pop(temp_max[0]) ;
			
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp.train.model ' + data_name+'-temp.train' ;
			subprocess.call (cmd.split()) ;
			m = load_model (data_name+'-temp.train.model') ;
			
			p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m) ;
			A_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m) ;
			A_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 3) :
		train_order = [i for i in range(len(train_x_copy))] ;
		for i in range (train_data_len-1) :
			print (i) ;
			temp_max = [0 , 0.0] ;
			for j in range (len(train_x_copy)) :
				train_x_copy2 = copy.deepcopy(train_x_copy) ;
				train_y_copy2 = copy.deepcopy(train_y_copy) ;
				
				train_x_copy2.pop(j) ;
				train_y_copy2.pop(j) ;
				
				write_svm_file (train_x_copy2 , train_y_copy2 , data_name+'-temp2.train') ;
				cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp.train.model ' + data_name+'-temp2.train' ;
				subprocess.call (cmd.split()) ;
				m2 = load_model (data_name+'-temp2.train.model') ;

				p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m2) ;
				if (p_acc[0] > temp_max[1]) :
					temp_max[1] = p_acc[0] ;
					temp_max[0] = j ;
			
			pop_list.append(train_order.pop(temp_max[0])) ;
			train_x_copy.pop(temp_max[0]) ;
			train_y_copy.pop(temp_max[0]) ;
			
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp.train.model ' + data_name+'-temp.train' ;
			subprocess.call (cmd.split()) ;
			m = load_model (data_name+'-temp.train.model') ;
			
			p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m) ;
			A_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m) ;
			A_out.append(p_acc[0]/base_acc_out) ;	
	
	elif (select_method == 4) :
		train_order = [i for i in range(len(train_x_copy))] ;
		for i in range (train_data_len-1) :
			print (i) ;
			temp_max = [0 , 0.0] ;
			for j in range (len(train_x_copy)) :
				train_x_copy2 = copy.deepcopy(train_x_copy) ;
				train_y_copy2 = copy.deepcopy(train_y_copy) ;
				
				train_x_copy2.pop(j) ;
				train_y_copy2.pop(j) ;
				
				write_svm_file (train_x_copy2 , train_y_copy2 , data_name+'-temp2.train') ;
				cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp.train.model ' + data_name+'-temp2.train' ;
				subprocess.call (cmd.split()) ;
				m2 = load_model (data_name+'-temp2.train.model') ;

				p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
				if (p_acc[0] > temp_max[1]) :
					temp_max[1] = p_acc[0] ;
					temp_max[0] = j ;
			
			pop_list.append(train_order.pop(temp_max[0])) ;
			train_x_copy.pop(temp_max[0]) ;
			train_y_copy.pop(temp_max[0]) ;
			
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp.train.model ' + data_name+'-temp.train' ;
			subprocess.call (cmd.split()) ;
			m = load_model (data_name+'-temp.train.model') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m) ;
			A_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m) ;
			A_out.append(p_acc[0]/base_acc_out) ;	
	
	return A_in , A_out , pop_list ;		

# ------------------------------------

train_file_name = sys.argv[1] ;
test_file_name = train_file_name + '.t' ;
pure_data_name = (train_file_name.split('/'))[len(train_file_name.split('/'))-1] ;

print (pure_data_name) ;

# ------------------------------------

train_y , train_x = svm_read_problem(train_file_name) ;
test_y , test_x = svm_read_problem(test_file_name) ;

train_x , train_y = check_zero (train_x , train_y) ;
test_x , test_y = check_zero (test_x , test_y) ;

write_svm_file (train_x , train_y , pure_data_name+'-temp.train') ;

train_data_len = len(train_y) ;
# train_data_len = 4 ;

# ------------------------------------

cmd = '../package/liblinear-incdec-2.01/train -s 2 -q ' + pure_data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
m = load_model (pure_data_name+'-temp.train.model') ;

p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
base_acc_in = p_acc_in[0] ;
p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;
base_acc_out = p_acc_out[0] ;
A_in_0 = [1.0 for i in range(train_data_len)] ;
A_out_0 = [1.0 for i in range(train_data_len)] ;

# ------------------------------------

# select_method = 1 ;
# A_in_1 , A_out_1 , pop_list_1 = run (pure_data_name , select_method , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('greedy_out_part') ;

select_method = 3 ;
A_in_3 , A_out_3 , pop_list_3 = run (pure_data_name , select_method , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('greedy_in_part') ;

select_method = 4 ;
A_in_4 , A_out_4 , pop_list_4 = run (pure_data_name , select_method , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('greedy_in_total') ;

# ------------------------------------

cmd = 'rm ' + pure_data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
cmd = 'rm ' + pure_data_name+'-temp.train.model' ;
subprocess.call (cmd.split()) ;

cmd = 'rm ' + pure_data_name+'-temp2.train' ;
subprocess.call (cmd.split()) ;
cmd = 'rm ' + pure_data_name+'-temp2.train.model' ;
subprocess.call (cmd.split()) ;

# ------------------------------------

query_num = np.arange(1.00 , 0.00 , -1.0/(1.0*train_data_len)) ;
plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4) ;
ax = plt.gca() ; 
ax.xaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
# ax.yaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('Acc rate') ;
plt.xlim(1.00 , 0.00) ;
# plt.ylim(0.8 , 1.2) ;
plt.grid () ;

plt.title(pure_data_name + '_in_' + ('%.3f' % base_acc_in)) ;
plt.plot(query_num, A_in_0, 'k', label='total') ;
# plt.plot(query_num, A_in_1, 'b', label='greedy_out') ;
plt.plot(query_num, A_in_3, 'r', label='greedy_in_part') ;
plt.plot(query_num, A_in_4, 'g', label='greedy_in_total') ;


plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_in' + '.png') ;

plt.cla() ;

# ----------------------------------------------------------

query_num = np.arange(1.00 , 0.00 , -1.0/(1.0*train_data_len)) ;
plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4) ;
ax = plt.gca() ; 
ax.xaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
# ax.yaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('Acc rate') ;
plt.xlim(1.00 , 0.00) ;
# plt.ylim(0.5 , 1.5) ;
plt.grid () ;

plt.title(pure_data_name + '_out_' + ('%.3f' % base_acc_out)) ;
plt.plot(query_num, A_out_0, 'k', label='total') ;
# plt.plot(query_num, A_out_1, 'b', label='greedy_out') ;
plt.plot(query_num, A_out_3, 'r', label='greedy_in_part') ;
plt.plot(query_num, A_out_4, 'g', label='greedy_in_total') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_out' + '.png') ;

plt.cla() ;

# ------------------------------------

# output_file_name = pure_data_name + '_greedy_out.pop' ;
# output_file = open (output_file_name , 'w') ;
# for p in pop_list_1 :
	# output_file.write ('%d\n' % p) ;
	
output_file_name = pure_data_name + '_greedy_in_part.pop' ;
output_file = open (output_file_name , 'w') ;
for p in pop_list_3 :
	output_file.write ('%d\n' % p) ;
	
output_file_name = pure_data_name + '_greedy_in_total.pop' ;
output_file = open (output_file_name , 'w') ;
for p in pop_list_4 :
	output_file.write ('%d\n' % p) ;
