import sys ;
import pickle ;
import numpy as np ;
from liblinearutil import * ;
from random import * ;
import copy ;
import subprocess ;
import time ;
from scipy.spatial import distance ;
from sklearn.svm import SVR ;
import math ;

import matplotlib ;
matplotlib.use('Agg') ;
import matplotlib.pyplot as plt ;
from matplotlib.ticker import MultipleLocator, FuncFormatter ;

# -------------------------------------------------------

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
	
# ------------------------------------------------------- 
	
train_file_name_origin = sys.argv[1] ;
train_file_name = train_file_name_origin + '.s1' ; 
test_file_name = train_file_name_origin + '.t' ;
pure_data_name = (train_file_name_origin.split('/'))[len(train_file_name_origin.split('/'))-1] ;

gr_file = open ('../../data/data_greedy/' + pure_data_name + '_99per_Eout.pickle' , 'rb') ;
_ , greedy_pop_list , E_out_gr , E_in_gr = pickle.load(gr_file) ;
gr_file.close() ;

pop_num = len(greedy_pop_list) ;

# -------------------------------------------------------

train_y , train_x = svm_read_problem(train_file_name) ;
test_y , test_x = svm_read_problem(test_file_name) ;

train_x , train_y = check_zero (train_x , train_y) ;
test_x , test_y = check_zero (test_x , test_y) ;

write_svm_file (train_x , train_y , pure_data_name+'-temp.train') ;

# -------------------------------------------------------

cmd = '../package/liblinear-incdec-2.01/train -s 2 -q ' + pure_data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
m = load_model (pure_data_name+'-temp.train.model') ;

p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
base_acc_in = p_acc_in[0] ;
p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;
base_acc_out = p_acc_out[0] ;

print (base_acc_in) ;

# -------------------------------------------------------

train_x_copy = copy.deepcopy(train_x) ;
train_y_copy = copy.deepcopy(train_y) ;
train_order = [i for i in range(len(train_x_copy))] ;

data_name = pure_data_name ;
write_svm_file (train_x_copy , train_y_copy , data_name+'-temp.train') ;
cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp.train.model ' + data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
m2 = load_model (data_name+'-temp.train.model') ;
p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m2) ;

right_gr_list = [0 for i in range(pop_num)] ;
decision_gr_list = [0 for i in range(pop_num)] ;
for i in range (pop_num) :
	temp_index = greedy_pop_list[i] ;
	if (p_label[temp_index] == train_y_copy[temp_index]) :
		right_gr_list[i] = 1 ;
	decision_gr_list[i] = max(p_val[temp_index]) ;
	
euclidean_dst_min_list = [0.0 for k in range(len(train_x_copy))] ;
euclidean_dst_avg_list = [0.0 for k in range(len(train_x_copy))] ;
class_dst_min_dict = dict() ; 
class_dst_avg_dict = dict() ; 
class_dst_min_dict_2 = dict() ;
class_dst_avg_dict_2 = dict() ;
for k in range(len(train_x_copy)) :
	temp_min_dst = -1.0 ;
	temp_list = list() ;
	for k2 in range(len(train_x_copy)) :
		if (k != k2 and train_y_copy[k] == train_y_copy[k2]) :
			temp = [train_x_copy[k] , train_x_copy[k2]] ;
			words = sorted(list(reduce(set.union, map(set, temp)))) ;
			feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
			dst = distance.euclidean(feats[0],feats[1]) ;
			if (temp_min_dst == -1.0 or dst < temp_min_dst) :
				temp_min_dst = dst ;
			temp_list.append(dst) ;
	euclidean_dst_min_list[k] = temp_min_dst ;	
	euclidean_dst_avg_list[k] = np.mean(temp_list) ;
	
	if (train_y_copy[k] not in class_dst_min_dict) :
		class_dst_min_dict[train_y_copy[k]] = list() ;
		class_dst_min_dict_2[train_y_copy[k]] = dict() ;
	if (train_y_copy[k] not in class_dst_avg_dict) :
		class_dst_avg_dict[train_y_copy[k]] = list() ;
		class_dst_avg_dict_2[train_y_copy[k]] = dict() ;
	class_dst_min_dict[train_y_copy[k]].append(euclidean_dst_min_list[k]) ;	
	class_dst_avg_dict[train_y_copy[k]].append(euclidean_dst_avg_list[k]) ;	

for label in class_dst_min_dict.keys() :
	class_dst_min_dict_2[label]['avg'] = np.mean(class_dst_min_dict[label]) ;
	class_dst_min_dict_2[label]['std'] = np.std(class_dst_min_dict[label]) ;

for label in class_dst_min_dict.keys() :
	class_dst_avg_dict_2[label]['avg'] = np.mean(class_dst_avg_dict[label]) ;
	class_dst_avg_dict_2[label]['std'] = np.std(class_dst_avg_dict[label]) ;
	
euclidean_dst_min_gr_list = [0.0 for i in range(pop_num)] ;
euclidean_dst_avg_gr_list = [0.0 for i in range(pop_num)] ;
label_gr_list = [0.0 for i in range(pop_num)] ;
for i in range (pop_num) :
	temp_index = greedy_pop_list[i] ;
	label_gr_list[i] = train_y_copy[temp_index] ;
	euclidean_dst_min_gr_list[i] = euclidean_dst_min_list[temp_index] ;
	euclidean_dst_avg_gr_list[i] = euclidean_dst_avg_list[temp_index] ;
	
# right_change_gr_list = [0 for i in range(pop_num)] ;
# decision_change_gr_list = [0 for i in range(pop_num)] ;
# for i in range (pop_num) :
	# temp_int = train_order.index(greedy_pop_list.pop(0)) ;
	
	# write_svm_file (train_x_copy , train_y_copy , data_name+'-temp.train') ;
	# cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp.train.model ' + data_name+'-temp.train' ;
	# subprocess.call (cmd.split()) ;
	# m2 = load_model (data_name+'-temp.train.model') ;
	# p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m2) ;
	
	# if (p_label[temp_int] == train_y_copy[temp_int]) :
		# right_change_gr_list[i] = 1 ;
	# decision_change_gr_list[i] = max(p_val[temp_int]) ;
	
	# train_order.pop(temp_int) ;
	# train_x_copy.pop(temp_int) ;
	# train_y_copy.pop(temp_int) ;

print ('Greedy_out') ;

# -------------------------------------------------------

dif_E_in_gr = [0.0 for i in range(pop_num)] ;
dif_E_out_gr = [0.0 for i in range(pop_num)] ;

for i in range (pop_num) :
	if (i == 0) :
		dif_E_in_gr[i] = E_in_gr[i] - base_acc_in ;
		dif_E_out_gr[i] = E_out_gr[i] - base_acc_out ;
	else :
		dif_E_in_gr[i] = E_in_gr[i] - E_in_gr[i-1] ;
		dif_E_out_gr[i] = E_out_gr[i] - E_out_gr[i-1] ;

positive_count = 0 ;
both_positive_count = 0 ;
right_count = 0 ;
dicision_list = list() ;
dst_min_list = list() ;
dst_avg_list = list() ;

for i in range (100) :
	if (dif_E_out_gr[i] > 0.0) :
		positive_count += 1 ;
		dicision_list.append (decision_gr_list[i]) ;
		temp_min_dst = (euclidean_dst_min_gr_list[i]-class_dst_min_dict_2[label_gr_list[i]]['avg'] ) / class_dst_min_dict_2[label_gr_list[i]]['std'] ;
		temp_avg_dst = (euclidean_dst_avg_gr_list[i]-class_dst_avg_dict_2[label_gr_list[i]]['avg'] ) / class_dst_avg_dict_2[label_gr_list[i]]['std'] ;
		dst_min_list.append (temp_min_dst) ;
		dst_avg_list.append (temp_avg_dst) ;
		
	if (dif_E_out_gr[i] > 0.0 and dif_E_in_gr[i] > 0.0) :
		both_positive_count += 1 ;		
	if (dif_E_out_gr[i] > 0.0 and right_gr_list[i] == 1) :
		right_count += 1 ;
	
# -------------------------------------------------------

print ('positive_count=%d' % positive_count) ;
print ('both_positive_count=%d' % both_positive_count) ;
print ('right_count=%d' % right_count) ;
print ('dicision_mean=%f' % np.mean(dicision_list)) ;
print ('dicision_std=%f' % np.std(dicision_list)) ;
print ('dst_min_mean=%f' % np.mean(dst_min_list)) ;
print ('dst_min_std=%f' % np.std(dst_min_list)) ;
print ('dst_avg_mean=%f' % np.mean(dst_avg_list)) ;
print ('dst_avg_std=%f' % np.std(dst_avg_list)) ;
print ('pop_num=%d' % pop_num) ;

# -------------------------------------------------------

cmd = 'rm ' + pure_data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
cmd = 'rm ' + pure_data_name+'-temp.train.model' ;
subprocess.call (cmd.split()) ;
