# python train.py train_file_name

from liblinearutil import * ;
import numpy as np ;
from random import * ;
import copy ;
import subprocess ;
import time ;
import sys ;
from scipy.spatial import distance ;
from sklearn.svm import SVR ;
import math ;
import pickle ;

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
	cmd = '../package/liblinear-incdec-2.01/train -s 2 -q ' + data_name+'-temp2.train' ;
	subprocess.call (cmd.split()) ;
	m = load_model (data_name+'-temp2.train.model') ;
	p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
	p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;

	# label , acc , val = predict(train_y_copy , train_x_copy , m , '-b 1') ;
	label , acc , val = predict(train_y_copy , train_x_copy , m) ;
	new_val = list(map(max , val)) ;
	new_val = list(map(abs , new_val)) ;
	train_x_hyper = [x for (v , x) in sorted(zip(new_val , train_x_copy)) ] ;
	train_y_hyper = [y for (v , y) in sorted(zip(new_val , train_y_copy)) ] ;
	label_hyper = [l for (v , l) in sorted(zip(new_val , label)) ] ;
	
	
	# new_val_2 = [0.0 for i in range(len(val))] ;
	# for i in range(len(val)) :
		# tmp_max = max(val[i]) ;
		# val[i].remove(tmp_max) ;
		# new_val_2[i] = tmp_max-max(val[i]) ;
	# train_x_hyper_2 = [x for (v , x) in sorted(zip(new_val_2 , train_x_copy)) ] ;
	# train_y_hyper_2 = [y for (v , y) in sorted(zip(new_val_2 , train_y_copy)) ] ;
	# label_hyper_2 = [l for (v , l) in sorted(zip(new_val_2 , label)) ] ;
	
	E_in = [p_acc_in[0]/base_acc_in] ;
	E_out = [p_acc_out[0]/base_acc_out] ;
	# m = train (train_y_copy , train_x_copy , '-s 0 -q') ;

	if (select_method == 1) :
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = randint (0,len(train_x_copy)-1) ;
				# temp_int = len(train_x_copy)-1 ;
				train_x_copy.pop(temp_int) ;
				train_y_copy.pop(temp_int) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			
			p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 2) :
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp_min_dst = 10000000.0 ;
			for k2 in range(len(train_x_copy)) :
				if (k != k2 and train_y_copy[k] == train_y_copy[k2]) :
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
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 3) :
		total_f = dict() ;
		for k in range(len(train_x_copy)) :
			if (train_y_copy[k] not in total_f) :
				total_f[train_y_copy[k]] = dict() ;
			for f in train_x_copy[k].keys() :
				if (f not in total_f[train_y_copy[k]]) :
					total_f[train_y_copy[k]][f] = list() ;
				total_f[train_y_copy[k]][f].append(train_x_copy[k][f]) ;
		
		for key in total_f.keys() :
			for f in total_f[key].keys() :
				total_f[key][f] = np.mean(total_f[key][f]) ;
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp = [train_x_copy[k] , total_f[train_y_copy[k]]] ;
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
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
			
	elif (select_method == 4) :
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp_min_dst = 10000000.0 ;
			for k2 in range(len(train_x_copy)) :
				if (k != k2 and train_y_copy[k] == train_y_copy[k2]) :
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
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 5) :
		total_f = dict() ;
		for k in range(len(train_x_copy)) :
			if (train_y_copy[k] not in total_f) :
				total_f[train_y_copy[k]] = dict() ;
			for f in train_x_copy[k].keys() :
				if (f not in total_f[train_y_copy[k]]) :
					total_f[train_y_copy[k]][f] = list() ;
				total_f[train_y_copy[k]][f].append(train_x_copy[k][f]) ;
		for key in total_f.keys() :
			for f in total_f[key].keys() :
				total_f[key][f] = np.mean(total_f[key][f]) ;
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp = [train_x_copy[k] , total_f[train_y_copy[k]]] ;
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
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			
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
		
		temp_dict = dict() ;
		temp_u = dict() ;
		C_matrix_inv = dict() ;
		for k in range(len(train_x_copy)) :
			if (train_y_copy[k] not in temp_dict) :
				temp_dict[train_y_copy[k]] = list() ;
			temp_dict[train_y_copy[k]].append (train_x_copy[k]) ;
			
		for label in temp_dict.keys() :
			total_list = [[0.0 for l in range(temp_max)] for k in range(len(temp_dict[label]))] ;
			for k in range(len(temp_dict[label])) :
				for f in temp_dict[label][k].keys() :
					total_list[k][f-1] = temp_dict[label][k][f] ;
			
			temp_u[label] = [0.0 for l in range(temp_max)] ;
			total_list_t = np.transpose (total_list) ;
			for l in range(temp_max) :
				temp_u[label][l] = np.mean(total_list_t[l]) ;
			
			C_matrix = np.cov(total_list_t) ;
			try:
				C_matrix_inv[label] = np.linalg.inv(C_matrix) ;
			except np.linalg.LinAlgError:
				E_in = [noise_acc_in/base_acc_in for i in range(data_num_size+1)] ;
				E_out = [noise_acc_in/base_acc_in for i in range(data_num_size+1)] ;
				return E_in , E_out ;
		
		dst_list = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp_list = [0.0 for l in range(temp_max)] ;
			for f in train_x_copy[k].keys() :
				temp_list[f-1] = train_x_copy[k][f] ;
			dst_list[k] = distance.mahalanobis(temp_list,temp_u[train_y_copy[k]],C_matrix_inv[train_y_copy[k]]) ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_pop_index = dst_list.index(max(dst_list)) ;
				dst_list.pop(temp_pop_index) ;
				train_x_copy.pop(temp_pop_index) ;
				train_y_copy.pop(temp_pop_index) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 7) :
		train_order = [i for i in range(len(train_x_copy))] ;
		pop_list = [] ;
		pop_file = open ('../../data/data_greedy/' + data_name + '_greedy_out.pop' , 'r') ;
		while True :
			temp_index = pop_file.readline() ;
			if not temp_index :
				break ;
			temp_index = temp_index[:len(temp_index)-1] ;
			pop_list.append (int(temp_index)) ;
		
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = train_order.index(pop_list.pop(0)) ;
				# temp_int = len(train_x_copy)-1 ;
				train_order.pop(temp_int) ;
				train_x_copy.pop(temp_int) ;
				train_y_copy.pop(temp_int) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
			
	elif (select_method == 8) :
		train_order = [i for i in range(len(train_x_copy))] ;
		pop_list = [] ;
		pop_file = open ('../../data/data_greedy/' + data_name+ '_greedy_out.pop' , 'r') ;
		count = 0 ;
		while True :
			temp_index = pop_file.readline() ;
			if not temp_index :
				break ;
			count += 1 ;
			if (count == 200) :
				shuffle(pop_list) ;
			temp_index = temp_index[:len(temp_index)-1] ;
			pop_list.append (int(temp_index)) ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = train_order.index(pop_list.pop(0)) ;
				# temp_int = len(train_x_copy)-1 ;
				train_order.pop(temp_int) ;
				train_x_copy.pop(temp_int) ;
				train_y_copy.pop(temp_int) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 9) :
		train_order = [i for i in range(len(train_x_copy))] ;
		pop_list = [] ;
		pop_file = open ('../../data/data_greedy/' + data_name+ '_greedy_out.pop' , 'r') ;
		while True :
			temp_index = pop_file.readline() ;
			if not temp_index :
				break ;
			temp_index = temp_index[:len(temp_index)-1] ;
			pop_list.append (int(temp_index)) ;
		
		dst_list_man_pair = [0.0 for k in range(len(train_x_copy))] ;
		dst_list_euc_pair = [0.0 for k in range(len(train_x_copy))] ;
		for k in range(len(train_x_copy)) :
			temp_min_dst = 10000000.0 ;
			temp_min_dst2 = 10000000.0 ;
			for k2 in range(len(train_x_copy)) :
				if (k != k2 and train_y_copy[k] == train_y_copy[k2]) :
					temp = [train_x_copy[k] , train_x_copy[k2]] ;
					words = sorted(list(reduce(set.union, map(set, temp)))) ;
					feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
					dst = distance.cityblock(feats[0],feats[1]) ;
					dst2 = distance.euclidean(feats[0],feats[1]) ;
					if (dst < temp_min_dst) :
						temp_min_dst = dst ;
					if (dst2 < temp_min_dst2) :
						temp_min_dst2 = dst2 ;
			dst_list_man_pair[k] = temp_min_dst ;
			dst_list_euc_pair[k] = temp_min_dst2 ;
		
		train_x_copy2 = copy.deepcopy(train_x) ;
		train_y_copy2 = copy.deepcopy(train_y) ;
		re_train_x = [] ;
		re_train_y = [] ;
		for i in range(len(pop_list)) :
			temp_int = train_order.index(pop_list.pop(0)) ;
			
			re_train_x.append(list()) ;
			re_train_x[i].append(dst_list_man_pair.pop(temp_int)) ;
			re_train_x[i].append(dst_list_euc_pair.pop(temp_int)) ;
			
			train_order.pop(temp_int) ;
			train_x_copy2.pop(temp_int) ;
			train_y_copy2.pop(temp_int) ;
			m2 = train(train_y_copy2 , train_x_copy2 , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			re_train_y.append((p_acc[0]/base_acc_out) - 1.0) ;
		clf = SVR() ;
		clf.fit(re_train_x , re_train_y) ;
		p_label_list = clf.predict (re_train_x) ;
		# print (clf.score(re_train_x , re_train_y)) ;
		# print (p_label_list) ;
		p_label_list = list(p_label_list) ;
		sum = 0.0 ;
		for i in range(len(re_train_y)) :
			sum += (re_train_y[i]-p_label_list[i])*(re_train_y[i]-p_label_list[i]) ;
		rmse_score = math.sqrt(sum/len(re_train_y)) ;
		print (rmse_score) ;
		
		train_order = [i for i in range(len(train_x_copy))] ;
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = p_label_list.index(max(p_label_list)) ;
				# print (temp_int) ;
				# print (train_order.pop(temp_int)) ;
				p_label_list.pop(temp_int) ;
				train_x_copy.pop(temp_int) ;
				train_y_copy.pop(temp_int) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 10) :
		for i in range (data_num_size) :
			for j in range (base_size) :
				# temp_int = new_val.index(max(new_val)) ;
				# new_val.pop(temp_int) ;
				train_x_hyper.pop() ;
				train_y_hyper.pop() ;

			write_svm_file (train_x_hyper , train_y_hyper , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper , train_x_hyper , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 11) :
		for i in range (data_num_size) :
			j = 0 ;
			for k in range (len(train_x_hyper)-1 , -1 , -1) :
				if (label_hyper[k] != train_y_hyper[k]) :
					train_x_hyper.pop(k) ;
					train_y_hyper.pop(k) ;
					label_hyper.pop(k) ;
					j += 1 ;
				if (j >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					train_x_hyper.pop(0) ;
					train_y_hyper.pop(0) ;
					label_hyper.pop(0) ;
			
			write_svm_file (train_x_hyper , train_y_hyper , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper , train_x_hyper , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;		
	
	elif (select_method == 12) :
		for i in range (data_num_size) :
			for j in range (base_size) :
				train_x_hyper.pop(0) ;
				train_y_hyper.pop(0) ;

			write_svm_file (train_x_hyper , train_y_hyper , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper , train_x_hyper , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 13) :
		for i in range (data_num_size) :
			for j in range (base_size) :
				train_x_hyper.pop() ;
				train_y_hyper.pop() ;

			write_svm_file (train_x_hyper , train_y_hyper , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper , train_x_hyper , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 14) :
		for i in range (data_num_size) :
			j = 0 ;
			for k in range (len(train_x_hyper)-1 , -1 , -1) :
				if (label_hyper[k] == train_y_hyper[k]) :
					train_x_hyper.pop(k) ;
					train_y_hyper.pop(k) ;
					label_hyper.pop(k) ;
					j += 1 ;
				if (j >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					train_x_hyper.pop() ;
					train_y_hyper.pop() ;
					label_hyper.pop() ;

			write_svm_file (train_x_hyper , train_y_hyper , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper , train_x_hyper , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 15) :
		pop_list = [] ;
		train_order = [i for i in range(len(train_x_copy))] ;
		for i in range (data_num_size) :
			j = 0 ;
			for k in range (len(train_x_hyper)-1 , -1 , -1) :
				if (label_hyper[k] != train_y_hyper[k]) :
					pop_list.append(train_order.pop(k)) ;
					train_x_hyper.pop(k) ;
					train_y_hyper.pop(k) ;
					label_hyper.pop(k) ;
					j += 1 ;
				if (j >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					pop_list.append(train_order.pop()) ;
					train_x_hyper.pop() ;
					train_y_hyper.pop() ;
					label_hyper.pop() ;

			write_svm_file (train_x_hyper , train_y_hyper , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper , train_x_hyper , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
		fi = open(data_name+'_LC_HW.pickle' , 'wb') ;
		pickle.dump(pop_list , fi) ;
		fi.close() ;
		
	elif (select_method == 16) :
		for i in range (data_num_size) :
			j = 0 ;
			while True :
				j2 = 0 ;
				for k in range (len(train_x_hyper)) :
					if (label_hyper[k] == train_y_hyper[k]) :
						train_x_hyper.pop(k) ;
						train_y_hyper.pop(k) ;
						label_hyper.pop(k) ;
						j += 1 ;
						break ;
					elif (k == len(train_x_hyper)-1) :
						j2 = base_size ;
				if (j+j2 >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					train_x_hyper.pop(0) ;
					train_y_hyper.pop(0) ;
					label_hyper.pop(0) ;

			write_svm_file (train_x_hyper , train_y_hyper , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper , train_x_hyper , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
			
	elif (select_method == 17) :
		for i in range (data_num_size) :
			j = 0 ;
			while True :
				j2 = 0 ;
				for k in range (len(train_x_hyper)) :
					if (label_hyper[k] != train_y_hyper[k]) :
						train_x_hyper.pop(k) ;
						train_y_hyper.pop(k) ;
						label_hyper.pop(k) ;
						j += 1 ;
						break ;
					elif (k == len(train_x_hyper)-1) :
						j2 = base_size ;
				if (j+j2 >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					train_x_hyper.pop(0) ;
					train_y_hyper.pop(0) ;
					label_hyper.pop(0) ;

			write_svm_file (train_x_hyper , train_y_hyper , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper , train_x_hyper , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 18) :
		for i in range (data_num_size) :
			for j in range (base_size) :
				train_x_hyper_2.pop(0) ;
				train_y_hyper_2.pop(0) ;

			write_svm_file (train_x_hyper_2 , train_y_hyper_2 , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper_2 , train_x_hyper_2 , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 19) :
		for i in range (data_num_size) :
			for j in range (base_size) :
				train_x_hyper_2.pop() ;
				train_y_hyper_2.pop() ;

			write_svm_file (train_x_hyper_2 , train_y_hyper_2 , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper_2 , train_x_hyper_2 , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 20) :
		for i in range (data_num_size) :
			j = 0 ;
			for k in range (len(train_x_hyper_2)-1 , -1 , -1) :
				if (label_hyper_2[k] == train_y_hyper_2[k]) :
					train_x_hyper_2.pop(k) ;
					train_y_hyper_2.pop(k) ;
					label_hyper_2.pop(k) ;
					j += 1 ;
				if (j >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					train_x_hyper_2.pop() ;
					train_y_hyper_2.pop() ;
					label_hyper_2.pop() ;

			write_svm_file (train_x_hyper_2 , train_y_hyper_2 , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper_2 , train_x_hyper_2 , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 21) :
		for i in range (data_num_size) :
			j = 0 ;
			for k in range (len(train_x_hyper_2)-1 , -1 , -1) :
				if (label_hyper_2[k] != train_y_hyper_2[k]) :
					train_x_hyper_2.pop(k) ;
					train_y_hyper_2.pop(k) ;
					label_hyper_2.pop(k) ;
					j += 1 ;
				if (j >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					train_x_hyper_2.pop() ;
					train_y_hyper_2.pop() ;
					label_hyper_2.pop() ;

			write_svm_file (train_x_hyper_2 , train_y_hyper_2 , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper_2 , train_x_hyper_2 , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 22) :
		train_x_hyper_2.reverse() ;
		train_y_hyper_2.reverse() ;
		label_hyper_2.reverse() ;
		for i in range (data_num_size) :
			j = 0 ;
			for k in range (len(train_x_hyper_2)-1 , -1 , -1) :
				if (label_hyper_2[k] == train_y_hyper_2[k]) :
					train_x_hyper_2.pop(k) ;
					train_y_hyper_2.pop(k) ;
					label_hyper_2.pop(k) ;
					j += 1 ;
				if (j >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					train_x_hyper_2.pop() ;
					train_y_hyper_2.pop() ;
					label_hyper_2.pop() ;

			write_svm_file (train_x_hyper_2 , train_y_hyper_2 , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper_2 , train_x_hyper_2 , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
			
	elif (select_method == 23) :
		train_x_hyper_2.reverse() ;
		train_y_hyper_2.reverse() ;
		label_hyper_2.reverse() ;
		for i in range (data_num_size) :
			j = 0 ;
			for k in range (len(train_x_hyper_2)-1 , -1 , -1) :
				if (label_hyper_2[k] != train_y_hyper_2[k]) :
					train_x_hyper_2.pop(k) ;
					train_y_hyper_2.pop(k) ;
					label_hyper_2.pop(k) ;
					j += 1 ;
				if (j >= base_size) :
					break ;
			if (j < base_size) :
				for k in range (j , base_size) :
					train_x_hyper_2.pop() ;
					train_y_hyper_2.pop() ;
					label_hyper_2.pop() ;

			write_svm_file (train_x_hyper_2 , train_y_hyper_2 , data_name+'-'+'temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 0 -q') ;
			p_label , p_acc , p_val = predict(train_y_hyper_2 , train_x_hyper_2 , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	
	elif (select_method == 24) :
		greedy_file_name = '../../data/data_greedy/' + data_name + '_99per_Eout.pickle' ;
		fi = open(greedy_file_name , 'rb') ;
		_ , pop_list , d_E_out , d_E_in = pickle.load(fi) ;
		fi.close() ;
		
		train_order = [i for i in range(len(train_x_copy))] ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = train_order.index(pop_list.pop(0)) ;
				train_order.pop(temp_int) ;
				train_x_copy.pop(temp_int) ;
				train_y_copy.pop(temp_int) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;

			p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;	
	
	elif (select_method == 25) :
		write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
		cmd = '../package/liblinear-incdec-2.01/train -s 0 -q ' + data_name+'-temp2.train' ;
		subprocess.call (cmd.split()) ;
		m2 = load_model (data_name+'-temp2.train.model') ;
		p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
		E_in[0] = p_acc[0]/base_acc_in ;
		p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
		E_out[0] = p_acc[0]/base_acc_out ;
		
		
		greedy_file_name = '../../data/data_greedy/' + data_name + '_99per_Eout.pickle' ;
		fi = open(greedy_file_name , 'rb') ;
		_ , pop_list , d_E_out , d_E_in = pickle.load(fi) ;
		fi.close() ;
		
		train_order = [i for i in range(len(train_x_copy))] ;

		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = train_order.index(pop_list.pop(0)) ;
				train_order.pop(temp_int) ;
				train_x_copy.pop(temp_int) ;
				train_y_copy.pop(temp_int) ;
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 0 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;

			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;	
	
	elif (select_method == 26) :
		greedy_file_name = '../../data/data_greedy/' + data_name + '_99per_Eout.pickle' ;
		fi = open(greedy_file_name , 'rb') ;
		_ , pop_list , d_E_out , d_E_in = pickle.load(fi) ;
		fi.close() ;
		
		pop_num = len(pop_list) ;
		dif_E_out_gr = [0.0 for i in range(pop_num)] ;
		for i in range (pop_num) :
			if (i == 0) :
				dif_E_out_gr[i] = d_E_out[i] - base_acc_out ;
			else :
				dif_E_out_gr[i] = d_E_out[i] - d_E_out[i-1] ;
		
		impove_list = list() ;
		for i in range (pop_num) :
			if (dif_E_out_gr[i] > 0.0) :
				impove_list.append(pop_list[i]) ;
		shuffle (impove_list) ;
		
		train_order = [i for i in range(len(train_x_copy))] ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				if (len(impove_list) > 0) :
					temp_int = train_order.index(impove_list.pop(0)) ;
					train_order.pop(temp_int) ;
					train_x_copy.pop(temp_int) ;
					train_y_copy.pop(temp_int) ;
				else :
					temp_int = randint(0,len(train_x_copy)-1) ;
					train_order.pop(temp_int) ;
					train_x_copy.pop(temp_int) ;
					train_y_copy.pop(temp_int) ;
					
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;

			p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;	
	
	elif (select_method == 27) :
		greedy_file_name = '../../data/data_greedy/' + data_name + '_99per_Eout.pickle' ;
		fi = open(greedy_file_name , 'rb') ;
		_ , pop_list , d_E_out , d_E_in = pickle.load(fi) ;
		fi.close() ;
		
		pop_num = len(pop_list) ;
		dif_E_out_gr = [0.0 for i in range(pop_num)] ;
		for i in range (pop_num) :
			if (i == 0) :
				dif_E_out_gr[i] = d_E_out[i] - base_acc_out ;
			else :
				dif_E_out_gr[i] = d_E_out[i] - d_E_out[i-1] ;
		
		impove_list = list() ;
		for i in range (200) :
			impove_list.append(pop_list[i]) ;
		for i in range (200) :
			pop_list.pop(0) ;
		shuffle (impove_list) ;
		
		train_order = [i for i in range(len(train_x_copy))] ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				if (len(impove_list) > 0) :
					temp_int = train_order.index(impove_list.pop(0)) ;
					train_order.pop(temp_int) ;
					train_x_copy.pop(temp_int) ;
					train_y_copy.pop(temp_int) ;
				else :
					temp_int = train_order.index(pop_list.pop(0)) ;
					train_order.pop(temp_int) ;
					train_x_copy.pop(temp_int) ;
					train_y_copy.pop(temp_int) ;
					
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;

			p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;	
	
	elif (select_method == 28) :
		greedy_file_name = '../../data/data_greedy/' + data_name + '_99per_Eout.pickle' ;
		fi = open(greedy_file_name , 'rb') ;
		_ , pop_list , d_E_out , d_E_in = pickle.load(fi) ;
		fi.close() ;
		
		pop_num = len(pop_list) ;
		dif_E_out_gr = [0.0 for i in range(pop_num)] ;
		for i in range (pop_num) :
			if (i == 0) :
				dif_E_out_gr[i] = d_E_out[i] - base_acc_out ;
			else :
				dif_E_out_gr[i] = d_E_out[i] - d_E_out[i-1] ;
		
		impove_list = list() ;
		unimprove_list = list() ;
		for i in range (pop_num) :
			if (dif_E_out_gr[i] > 0.0) :
				impove_list.append(pop_list[i]) ;
			else :
				unimprove_list.append(pop_list[i]) ;
		
		train_order = [i for i in range(len(train_x_copy))] ;
		
		for i in range (data_num_size) :
			for j in range (base_size) :
				if (len(impove_list) > 0) :
					temp_int = train_order.index(impove_list.pop(0)) ;
					train_order.pop(temp_int) ;
					train_x_copy.pop(temp_int) ;
					train_y_copy.pop(temp_int) ;
				else :
					temp_int = train_order.index(unimprove_list.pop(0)) ;
					train_order.pop(temp_int) ;
					train_x_copy.pop(temp_int) ;
					train_y_copy.pop(temp_int) ;
					
			write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
			cmd = '../package/liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;

			p_label , p_acc , p_val = predict(train_y_copy , train_x_copy , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;	
	
	return E_in , E_out ;		

# ------------------------------------

train_file_name_origin = sys.argv[1] ;

train_file_name = train_file_name_origin + '.s1' ; 
test_file_name = train_file_name_origin + '.t' ;
pure_data_name = (train_file_name_origin.split('/'))[len(train_file_name_origin.split('/'))-1] ;
data_num_size = 100-1 ;

print (pure_data_name) ;

# ------------------------------------

train_y , train_x = svm_read_problem(train_file_name) ;
test_y , test_x = svm_read_problem(test_file_name) ;

train_x , train_y = check_zero (train_x , train_y) ;
test_x , test_y = check_zero (test_x , test_y) ;

base_size = int(len(train_y)*(1.00-0.01)/data_num_size) ;
# base_size = 1 ;

write_svm_file (train_x , train_y , pure_data_name+'-temp.train') ;

# ------------------------------------

cmd = '../package/liblinear-incdec-2.01/train -s 2 -q ' + pure_data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
m = load_model (pure_data_name+'-temp.train.model') ;

p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
base_acc_in = p_acc_in[0] ;
p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;
base_acc_out = p_acc_out[0] ;
E_in_0 = [1.0 for i in range(data_num_size+1)] ;
E_out_0 = [1.0 for i in range(data_num_size+1)] ;

# ------------------------------------

select_method = 1 ;
E_in_1 , E_out_1 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('random') ;

# select_method = 2 ;
# E_in_2 , E_out_2 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Euclidean distance each other') ;

# select_method = 3 ;
# E_in_3 , E_out_3 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Euclidean distance mean') ;

# select_method = 4 ;
# E_in_4 , E_out_4 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Manhattan distance each other') ;

# select_method = 5 ;
# E_in_5 , E_out_5 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Manhattan distance mean') ;

# select_method = 6 ;
# E_in_6 , E_out_6 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Mahalanobis distance mean') ;

# select_method = 7 ;
# E_in_7 , E_out_7 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Greedy out') ;

# select_method = 8 ;
# E_in_8 , E_out_8 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Greedy out shuffle 200') ;

# select_method = 9 ;
# E_in_9 , E_out_9 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Regression') ;

# select_method = 10 ;
# E_in_10 , E_out_10 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('hyper high') ;

# select_method = 11 ;
# E_in_11 , E_out_11 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('wrong hyper high') ;

# select_method = 12 ;
# E_in_12 , E_out_12 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('LC') ;

# select_method = 13 ;
# E_in_13 , E_out_13 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('LC_Reverse') ;

# select_method = 14 ;
# E_in_14 , E_out_14 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('LC_H_C') ;

# select_method = 15 ;
# E_in_15 , E_out_15 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('LC_H_W') ;

# select_method = 16 ;
# E_in_16 , E_out_16 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('LC_L_C') ;

# select_method = 17 ;
# E_in_17 , E_out_17 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('LC_L_W') ;

# select_method = 18 ;
# E_in_18 , E_out_18 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('BT') ;

# select_method = 19 ;
# E_in_19 , E_out_19 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('BT_Reverse') ;

# select_method = 20 ;
# E_in_20 , E_out_20 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('BT_H_C') ;

# select_method = 21 ;
# E_in_21 , E_out_21 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('BT_H_W') ;

# select_method = 22 ;
# E_in_22 , E_out_22 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('BT_L_C') ;

# select_method = 23 ;
# E_in_23 , E_out_23 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('BT_L_W') ;

select_method = 24 ;
E_in_24 , E_out_24 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Greedy_out_2') ;

# select_method = 25 ;
# E_in_25 , E_out_25 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Greedy_out_0') ;

# select_method = 26 ;
# E_in_26 , E_out_26 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Greedy_random_improve') ;

# select_method = 27 ;
# E_in_27 , E_out_27 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
# print ('Greedy_random_200') ;

select_method = 28 ;
E_in_28 , E_out_28 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Greedy_improve') ;

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

query_num = np.arange(1.00 , 0.00 , -0.01) ;
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
plt.plot(query_num, E_in_0, 'k', label='total') ;
plt.plot(query_num, E_in_1, 'b', label='random') ;
# plt.plot(query_num, E_in_2, 'r', label='Euc pair') ;
# plt.plot(query_num, E_in_3, 'g', label='Euc mean') ;
# plt.plot(query_num, E_in_4, 'c', label='Man pair') ;
# plt.plot(query_num, E_in_5, 'm', label='Man mean') ;
# plt.plot(query_num, E_in_6, 'y', label='Mah mean') ;
# plt.plot(query_num, E_in_7, 'grey', label='Gre out') ;
# plt.plot(query_num, E_in_8, 'b', label='Gre out Sh 200') ;
# plt.plot(query_num, E_in_9, 'b', label='Regression') ;
# plt.plot(query_num, E_in_10, 'g', label='hyper high') ;
# plt.plot(query_num, E_in_11, 'm', label='wrong hyper high') ;
# plt.plot(query_num, E_in_12, 'b', label='LC') ;
# plt.plot(query_num, E_in_13, 'r', label='LC_R') ;
# plt.plot(query_num, E_in_14, 'g', label='LC_HC') ;
# plt.plot(query_num, E_in_15, 'c', label='LC_HW') ;
# plt.plot(query_num, E_in_16, 'm', label='LC_LC') ;
# plt.plot(query_num, E_in_17, 'y', label='LC_LW') ;
# plt.plot(query_num, E_in_18, 'b', label='BT') ;
# plt.plot(query_num, E_in_19, 'r', label='BT_R') ;
# plt.plot(query_num, E_in_20, 'g', label='BT_HC') ;
# plt.plot(query_num, E_in_21, 'c', label='BT_HW') ;
# plt.plot(query_num, E_in_22, 'm', label='BT_LC') ;
# plt.plot(query_num, E_in_23, 'y', label='BT_LW') ;
plt.plot(query_num, E_in_24, 'grey', label='Ger out') ;
# plt.plot(query_num, E_in_25, 'b', label='Ger out 0') ;
# plt.plot(query_num, E_in_26, 'g', label='Ger ran imp') ;
# plt.plot(query_num, E_in_27, 'm', label='Ger ran 200') ;
plt.plot(query_num, E_in_28, 'm', label='Ger imp') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_in' + '.png') ;

plt.cla() ;

# ----------------------------------------------------------

query_num = np.arange(1.00 , 0.00 , -0.01) ;
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
plt.plot(query_num, E_out_0, 'k', label='total') ;
plt.plot(query_num, E_out_1, 'b', label='random') ;
# plt.plot(query_num, E_out_2, 'r', label='Euc pair') ;
# plt.plot(query_num, E_out_3, 'g', label='Euc mean') ;
# plt.plot(query_num, E_out_4, 'c', label='Man pair') ;
# plt.plot(query_num, E_out_5, 'm', label='Man mean') ;
# plt.plot(query_num, E_out_6, 'y', label='Mah mean') ;
# plt.plot(query_num, E_out_7, 'grey', label='Gre out') ;
# plt.plot(query_num, E_out_8, 'b', label='Gre out Sh 200') ;
# plt.plot(query_num, E_out_9, 'b', label='Regression') ;
# plt.plot(query_num, E_out_10, 'g', label='hyper high') ;
# plt.plot(query_num, E_out_11, 'm', label='wrong hyper high') ;
# plt.plot(query_num, E_out_12, 'b', label='LC') ;
# plt.plot(query_num, E_out_13, 'r', label='LC_R') ;
# plt.plot(query_num, E_out_14, 'g', label='LC_HC') ;
# plt.plot(query_num, E_out_15, 'c', label='LC_HW') ;
# plt.plot(query_num, E_out_16, 'm', label='LC_LC') ;
# plt.plot(query_num, E_out_17, 'y', label='LC_LW') ;
# plt.plot(query_num, E_out_18, 'b', label='BT') ;
# plt.plot(query_num, E_out_19, 'r', label='BT_R') ;
# plt.plot(query_num, E_out_20, 'g', label='BT_HC') ;
# plt.plot(query_num, E_out_21, 'c', label='BT_HW') ;
# plt.plot(query_num, E_out_22, 'm', label='BT_LC') ;
# plt.plot(query_num, E_out_23, 'y', label='BT_LW') ;
plt.plot(query_num, E_out_24, 'grey', label='Ger out') ;
# plt.plot(query_num, E_out_25, 'b', label='Ger out 0') ;
# plt.plot(query_num, E_out_26, 'g', label='Ger ran imp') ;
# plt.plot(query_num, E_out_27, 'm', label='Ger ran 200') ;
plt.plot(query_num, E_out_28, 'm', label='Ger imp') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_out' + '.png') ;

plt.cla() ;

# ------------------------------------