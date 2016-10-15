# python train.py train_file_name noise_mode

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

def run (data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) :
	train_x_copy = copy.deepcopy(train_x) ;
	train_y_copy = copy.deepcopy(train_y) ;
	
	write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
	cmd = '../../liblinear-incdec-2.01/train -s 2 -q ' + data_name+'-temp2.train' ;
	subprocess.call (cmd.split()) ;
	m = load_model (data_name+'-temp2.train.model') ;
	p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
	noise_acc_in = p_acc_in[0] ;
	p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;
	noise_acc_out = p_acc_out[0] ;
	
	E_in = [noise_acc_in/base_acc_in] ;
	E_out = [noise_acc_out/base_acc_out] ;
	# m = train (train_y_copy , train_x_copy , '-s 2 -q') ;
	
	if (select_method == 1) :
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = randint (0,len(train_x_copy)-1) ;
				# temp_int = len(train_x_copy)-1 ;
				train_x_copy.pop(temp_int) ;
				train_y_copy.pop(temp_int) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 2) :
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp_min_dst = 10000000.0 ;
			for k2 in range(len(train_x_copy)) :
				if (k != k2) :
					temp = [train_x_copy[k] , train_x_copy[k2]] ;
					words = sorted(list(reduce(set.union, map(set, temp)))) ;
					feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
					dst = distance.euclidean(feats[0],feats[1]) ;
					if (dst < temp_min_dst) :
						temp_min_dst = dst ;
			dst_list[k] = temp_min_dst ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_pop_index = dst_list.index(max(dst_list)) ;
				dst_list.pop(temp_pop_index) ;
				train_x_copy.pop(temp_pop_index) ;
				train_y_copy.pop(temp_pop_index) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 3) :
		total_f = dict() ;
		for k in range(len(train_x_copy)) :
			for f in train_x_copy[k].keys() :
				if (f not in total_f) :
					total_f[f] = list() ;
				total_f[f].append(train_x_copy[k][f]) ;
		for f in total_f.keys() :
			total_f[f] = np.mean(total_f[f]) ;
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp = [train_x_copy[k] , total_f] ;
			words = sorted(list(reduce(set.union, map(set, temp)))) ;
			feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
			dst_list[k] = distance.euclidean(feats[0],feats[1]) ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_pop_index = dst_list.index(max(dst_list)) ;
				dst_list.pop(temp_pop_index) ;
				train_x_copy.pop(temp_pop_index) ;
				train_y_copy.pop(temp_pop_index) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
			
	elif (select_method == 4) :
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp_min_dst = 10000000.0 ;
			for k2 in range(len(train_x_copy)) :
				if (k != k2) :
					temp = [train_x_copy[k] , train_x_copy[k2]] ;
					words = sorted(list(reduce(set.union, map(set, temp)))) ;
					feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
					dst = distance.cityblock(feats[0],feats[1]) ;
					if (dst < temp_min_dst) :
						temp_min_dst = dst ;
			dst_list[k] = temp_min_dst ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_pop_index = dst_list.index(max(dst_list)) ;
				dst_list.pop(temp_pop_index) ;
				train_x_copy.pop(temp_pop_index) ;
				train_y_copy.pop(temp_pop_index) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 5) :
		total_f = dict() ;
		for k in range(len(train_x_copy)) :
			for f in train_x_copy[k].keys() :
				if (f not in total_f) :
					total_f[f] = list() ;
				total_f[f].append(train_x_copy[k][f]) ;
		for f in total_f.keys() :
			total_f[f] = np.mean(total_f[f]) ;
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp = [train_x_copy[k] , total_f] ;
			words = sorted(list(reduce(set.union, map(set, temp)))) ;
			feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
			dst_list[k] = distance.cityblock(feats[0],feats[1]) ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_pop_index = dst_list.index(max(dst_list)) ;
				dst_list.pop(temp_pop_index) ;
				train_x_copy.pop(temp_pop_index) ;
				train_y_copy.pop(temp_pop_index) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 6) :
		temp_max = 0 ;
		for k in range(len(train_x_copy)) :
			for f in train_x_copy[k].keys() :
				if (f > temp_max) :
					temp_max = f ;

		total_list = [[0.0 for l in range(temp_max)] for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			for f in train_x_copy[k].keys() :
				total_list[k][f-1] = train_x_copy[k][f] ;
		
		temp_u = [0.0 for l in range(temp_max)] ;
		total_list_t = np.transpose (total_list) ;
		for l in range(temp_max) :
			temp_u[l] = np.mean(total_list_t[l]) ;
		
		C_matrix = np.cov(total_list_t) ;
		try:
			C_matrix_inv = np.linalg.inv(C_matrix) ;
		except np.linalg.LinAlgError:
			E_in = [noise_acc_in/base_acc_in for i in range(data_num_size+1)] ;
			E_out = [noise_acc_in/base_acc_in for i in range(data_num_size+1)] ;
			return E_in , E_out ;
		
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp_list = [0.0 for l in range(temp_max)] ;
			for f in train_x_copy[k].keys() :
				temp_list[f-1] = train_x_copy[k][f] ;
			dst_list[k] = distance.mahalanobis(temp_list,temp_u,C_matrix_inv) ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_pop_index = dst_list.index(max(dst_list)) ;
				dst_list.pop(temp_pop_index) ;
				train_x_copy.pop(temp_pop_index) ;
				train_y_copy.pop(temp_pop_index) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	return E_in , E_out ;		

# ------------------------------------

train_file_name = sys.argv[1] ;
test_file_name = train_file_name + '.t' ;
pure_data_name = (train_file_name.split('/'))[len(train_file_name.split('/'))-1] ;
data_num_size = 10 ;

print (pure_data_name) ;

# ------------------------------------

train_y , train_x = svm_read_problem(train_file_name) ;
test_y , test_x = svm_read_problem(test_file_name) ;

train_x , train_y = check_zero (train_x , train_y) ;
test_x , test_y = check_zero (test_x , test_y) ;

base_size = int(len(train_y)*0.1/data_num_size) + 1 ;
# base_size = 1 ;

write_svm_file (train_x , train_y , pure_data_name+'-temp.train') ;

# ------------------------------------

cmd = '../../liblinear-incdec-2.01/train -s 2 -q ' + pure_data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
m = load_model (pure_data_name+'-temp.train.model') ;

p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
base_acc_in = p_acc_in[0] ;
p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;
base_acc_out = p_acc_out[0] ;
E_in_0 = [1.0 for i in range(data_num_size+1)] ;
E_out_0 = [1.0 for i in range(data_num_size+1)] ;

# ------------------------------------

mode = sys.argv[2] ;
train_y , train_x = svm_read_problem(train_file_name+'.n'+mode) ;
train_x , train_y = check_zero (train_x , train_y) ;

base_size = int(len(train_y)*0.1/data_num_size) + 1 ;
write_svm_file (train_x , train_y , pure_data_name+'-temp.train') ;

cmd = '../../liblinear-incdec-2.01/train -s 2 -q ' + pure_data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
m = load_model (pure_data_name+'-temp.train.model') ;

p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
noise_acc_in = p_acc_in[0] ;
p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;
noise_acc_out = p_acc_out[0] ;
E_in_02 = [noise_acc_in/base_acc_in for i in range(data_num_size+1)] ;
E_out_02 = [noise_acc_out/base_acc_out for i in range(data_num_size+1)] ;

# ------------------------------------

# random
select_method = 1 ;
E_in_1 , E_out_1 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('random') ;
print (E_out_1) ;

select_method = 2 ;
E_in_2 , E_out_2 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Euclidean distance each other') ;
print (E_out_2) ;

select_method = 3 ;
E_in_3 , E_out_3 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Euclidean distance mean') ;
print (E_out_3) ;

select_method = 4 ;
E_in_4 , E_out_4 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Manhattan distance each other') ;
print (E_out_4) ;

select_method = 5 ;
E_in_5 , E_out_5 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Manhattan distance mean') ;
print (E_out_5) ;

select_method = 6 ;
E_in_6 , E_out_6 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Mahalanobis distance mean') ;
print (E_out_6) ;

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

query_num = np.arange(1.00 , 0.89 , -0.01) ;
plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4) ;
ax = plt.gca() ; 
ax.xaxis.set_major_locator( MultipleLocator(0.01) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
# ax.yaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('Acc rate') ;
plt.xlim(1.00 , 0.90) ;
# plt.ylim(0.8 , 1.2) ;
plt.grid () ;

plt.title(pure_data_name + '_in_' +  mode + '_' + ('%.3f' % base_acc_in)) ;
plt.plot(query_num, E_in_0, 'k', label='total') ;
plt.plot(query_num, E_in_02, 'r', label='noise') ;
plt.plot(query_num, E_in_1, 'bo--', label='random') ;
plt.plot(query_num, E_in_2, 'rv--', label='Euc pair') ;
plt.plot(query_num, E_in_3, 'g^--', label='Euc mean') ;
plt.plot(query_num, E_in_4, 'c*--', label='Man pair') ;
plt.plot(query_num, E_in_5, 'mx--', label='Man mean') ;
plt.plot(query_num, E_in_6, 'yx--', label='Mah mean') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_in_' + mode + '.png') ;

plt.cla() ;

# ----------------------------------------------------------

query_num = np.arange(1.00 , 0.89 , -0.01) ;
plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4) ;
ax = plt.gca() ; 
ax.xaxis.set_major_locator( MultipleLocator(0.01) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
# ax.yaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('Acc rate') ;
plt.xlim(1.00 , 0.90) ;
# plt.ylim(0.5 , 1.5) ;
plt.grid () ;

plt.title(pure_data_name + '_out_' + mode + '_' + ('%.3f' % base_acc_out)) ;
plt.plot(query_num, E_out_0, 'k', label='total') ;
plt.plot(query_num, E_out_02, 'r', label='noise') ;
plt.plot(query_num, E_out_1, 'bo--', label='random') ;
plt.plot(query_num, E_out_2, 'rv--', label='Euc pair') ;
plt.plot(query_num, E_out_3, 'g^--', label='Euc mean') ;
plt.plot(query_num, E_out_4, 'c*--', label='Man pair') ;
plt.plot(query_num, E_out_5, 'mx--', label='Man mean') ;
plt.plot(query_num, E_out_6, 'yx--', label='Mah mean') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_out_' + mode + '.png') ;

plt.cla() ;

# ------------------------------------