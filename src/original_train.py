# python train.py train_file_name

from liblinearutil import * ;
import numpy as np ;
from random import * ;
import copy ;
import subprocess ;
import time ;
import sys ;
from scipy.spatial import distance ;
import re ;
import math ;

import matplotlib ;
matplotlib.use('Agg') ;
import matplotlib.pyplot as plt ;
from matplotlib.ticker import MultipleLocator, FuncFormatter ;
K=1 ;
MinPts = 10 ;

# ------------------------------------

class LOF:

	parameter_k = 0 ;
	MinPts = 0 ;
	all_Kdist = [] ;# record k_distance for all the points
	all_Ldist = [] ; # record nearst neighbor distance for all the points
	all_Rdist = [] ;# record reached_distance for all the points
	all_Ird_dist = [] ;# record Ird for all the points
	all_neighborX_list = [] ;# record neighbor X index for all the points 
	all_neighborY_list = [] ;# record neighbor Y for all the points
	all_LOF_list = [] ; # record LOF for all the points
	
	train_x_copy= [] ;
	train_y_copy = [];

	def __init__(self, train_x_copy, train_y_copy, parameter_k , MinPts):
		self.train_x_copy = train_x_copy ; 
		self.train_y_copy = train_y_copy ;
		self.parameter_k = parameter_k ;
		self.MinPts = MinPts ; 
	def main_control(self):
		self.k_distance() ; 
		"""
		print("L_distance") ;
		for i in range(len(self.all_Ldist)):
			print(self.all_Ldist[i]) ;
			print('\n') ;
		"""
		self.Ird() ;
		"""
		print("IRD")
		for i in range(len(self.all_Ird_dist)):
			print(self.all_Ird_dist[i]) ;
			print('\n') ;
		"""
		self.LOF_cal();
		"""
		print("LOF") ;
		for i in range(len(self.all_LOF_list)):
			print(self.all_LOF_list[i]) ;
			print('\n') ;
		"""

	def k_distance(self):
		for k in range(len(self.train_x_copy)) :
			temp_Kdist = [] ;
			temp_Ldist = [] ;
			temp_neighborX_list = [] ;
			temp_neighborY_list = [] ;
			for k2 in range(len(self.train_x_copy)) :
				if ((k != k2) and (self.train_y_copy[k] == self.train_y_copy[k2])) :
					temp = [self.train_x_copy[k] , self.train_x_copy[k2]] ;
					words = sorted(list(reduce(set.union, map(set, temp)))) ;
					feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
					#dst = distance.euclidean(feats[0],feats[1]) ;
					dst = distance.cityblock(feats[0],feats[1]) ;

					if (len(temp_Ldist) >= self.MinPts and (max(temp_Ldist) - dst > 0)):
						top_Lindex = temp_Ldist.index(max(temp_Ldist)) ;
						temp_Ldist[top_Lindex] = dst ;
						temp_neighborX_list[top_Lindex] = k2 ;# record neighbor index 
						temp_neighborY_list[top_Lindex] = self.train_y_copy[k2] ;# record neighbor label
					elif (len(temp_Ldist) < self.MinPts):
						temp_Ldist.append(dst) ;
						temp_neighborX_list.append(k2) ;
						temp_neighborY_list.append(self.train_y_copy[k2]) ;

					if (len(temp_Kdist) >= self.parameter_k and max(temp_Kdist) - dst > 0):
						top_index = temp_Kdist.index(max(temp_Kdist)) ;
						temp_Kdist[top_index] = dst ;
					elif (len(temp_Kdist) < self.parameter_k):
						temp_Kdist.append(dst) ;

			self.all_Kdist.append(max(temp_Kdist)) ;
			self.all_Ldist.append(temp_Ldist) ;
			self.all_neighborX_list.append(temp_neighborX_list) ;
			self.all_neighborY_list.append(temp_neighborY_list) ;
			
	def Ird(self):
		for k in range(len(self.train_x_copy)) :
			temp_Rdist = []
			#if(k==1):
				#print("all_Ldist",self.all_Ldist[k]) ;
			for i in range(self.MinPts):				
				if(self.all_Ldist[k][i]<self.all_Kdist[k]):
					reached_distance = self.all_Kdist[k] ;
				else:
					reached_distance = self.all_Ldist[k][i] ;
				"""
				if(k==1):
					print("i:",i) ;
					print("all_Ldist:", self.all_Ldist[k]) ;
					print("all_Kdist:", self.all_Kdist[k]) ;
					print("reached_distance", reached_distance) ;
				"""

				temp_Rdist.append(reached_distance) ;
			self.all_Rdist.append(temp_Rdist) ;
		for k in range(len(self.train_x_copy)) :
			"""
			tempReachedDist = 0 ;
			for i in range(self.MinPts):
				if(self.train_y_copy[k] != self.all_neighborY_list[k][i]):
					tempReachedDist = tempReachedDist + (1/self.all_Rdist[k][i]) ;
				else:
					tempReachedDist = tempReachedDist - (1/self.all_Rdist[k][i]) ;
			avgReachedDist = tempReachedDist/self.MinPts ;
			"""

			avgReachedDist = sum(self.all_Rdist[k])/(float)(len(self.all_Rdist[k])) ;
			Ird_value = 1/avgReachedDist ;
			self.all_Ird_dist.append(Ird_value) ;
	def LOF_cal(self):
		all_Ird_ratio = [] ;
		for k in range(len(self.train_x_copy)) :
			temp_Ird_ratio = [] ;
			for i in range(self.MinPts):
				neighborX = self.all_neighborX_list[k][i] ;
				Ird_neighbor = self.all_Ird_dist[neighborX] ;
				Ird_itself = self.all_Ird_dist[k] ;
				ratio = Ird_neighbor/Ird_itself ;
				temp_Ird_ratio.append(ratio) ;
			LOF_value = sum(temp_Ird_ratio)/(float)(len(temp_Ird_ratio)) ;
			self.all_LOF_list.append(LOF_value) ;# same label LOF
class Different_Label_LOF:

	parameter_k = 0 ;
	MinPts = 0 ;
	all_Kdist = [] ;# record k_distance for all the points
	all_Ldist = [] ; # record nearst neighbor distance for all the points
	all_Rdist = [] ;# record reached_distance for all the points
	all_Ird_dist = [] ;# record Ird for all the points
	all_neighborX_list = [] ;# record neighbor X index for all the points 
	all_neighborY_list = [] ;# record neighbor Y for all the points
	all_LOF_list = [] ; # record LOF for all the points
	same_neighborX_list = [] ;
	same_neighborY_list = [] ;
	same_Ird_dist = [] ;
	
	train_x_copy= [] ;
	train_y_copy = [];

	def __init__(self, train_x_copy, train_y_copy, parameter_k , MinPts):
		self.train_x_copy = train_x_copy ; 
		self.train_y_copy = train_y_copy ;
		self.parameter_k = parameter_k ;
		self.MinPts = MinPts ; 
		LOF_point = LOF(train_x_copy, train_y_copy, parameter_k, MinPts) ; 
		LOF_point.main_control() ; 
		self.same_neighborX_list = LOF_point.all_neighborX_list ;
		self.same_neighborY_list = LOF_point.all_neighborY_list ;
		self.same_Ird_dist = LOF_point.all_Ird_dist ;
	def main_control(self):
		self.k_distance() ; 
		"""
		print("L_distance") ;
		for i in range(len(self.all_Ldist)):
			print(self.all_Ldist[i]) ;
			print('\n') ;
		"""
		self.Ird() ;
		"""
		print("IRD")
		for i in range(len(self.all_Ird_dist)):
			print(self.all_Ird_dist[i]) ;
			print('\n') ;
		"""
		self.LOF_cal();
		"""
		print("LOF") ;
		for i in range(len(self.all_LOF_list)):
			print(self.all_LOF_list[i]) ;
			print('\n') ;
		"""

	def k_distance(self):
		for k in range(len(self.train_x_copy)) :
			temp_Kdist = [] ;
			temp_Ldist = [] ;
			temp_neighborX_list = [] ;
			temp_neighborY_list = [] ;
			for k2 in range(len(self.train_x_copy)) :
				if (k != k2) :
					temp = [self.train_x_copy[k] , self.train_x_copy[k2]] ;
					words = sorted(list(reduce(set.union, map(set, temp)))) ;
					feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
					#dst = distance.euclidean(feats[0],feats[1]) ;
					dst = distance.cityblock(feats[0],feats[1]) ;

					if (len(temp_Ldist) >= self.MinPts and (max(temp_Ldist) - dst > 0)):
						top_Lindex = temp_Ldist.index(max(temp_Ldist)) ;
						temp_Ldist[top_Lindex] = dst ;
						temp_neighborX_list[top_Lindex] = k2 ;# record neighbor index 
						temp_neighborY_list[top_Lindex] = self.train_y_copy[k2] ;# record neighbor label
					elif (len(temp_Ldist) < self.MinPts):
						temp_Ldist.append(dst) ;
						temp_neighborX_list.append(k2) ;
						temp_neighborY_list.append(self.train_y_copy[k2]) ;

					if (len(temp_Kdist) >= self.parameter_k and max(temp_Kdist) - dst > 0):
						top_index = temp_Kdist.index(max(temp_Kdist)) ;
						temp_Kdist[top_index] = dst ;
					elif (len(temp_Kdist) < self.parameter_k):
						temp_Kdist.append(dst) ;

					
			self.all_Kdist.append(max(temp_Kdist)) ;
			self.all_Ldist.append(temp_Ldist) ;
			self.all_neighborX_list.append(temp_neighborX_list) ;
			self.all_neighborY_list.append(temp_neighborY_list) ;
	def Ird(self):
		for k in range(len(self.train_x_copy)) :
			temp_Rdist = []
			for i in range(self.MinPts):
				if(self.all_Ldist[k][i]<self.all_Kdist[k]):
					reached_distance = self.all_Kdist[k] ;
				else:
					reached_distance = self.all_Ldist[k][i] ;
				temp_Rdist.append(reached_distance) ;
			self.all_Rdist.append(temp_Rdist) ;
		for k in range(len(self.train_x_copy)) :
			#print(sum(self.all_Rdist[k])) ;
			avgReachedDist = sum(self.all_Rdist[k])/(float)(len(self.all_Rdist[k])) ;
			Ird_value = 1/avgReachedDist ;
			self.all_Ird_dist.append(Ird_value) ;
	def LOF_cal(self):
		all_Ird_ratio = [] ;
		for k in range(len(self.train_x_copy)) :
			temp_Ird_ratio = [] ;
			for i in range(self.MinPts):

				neighborX = self.all_neighborX_list[k][i] ;
				Ird_neighbor = self.same_Ird_dist[neighborX] ;
				#print("neighborX", neighborX) ;
				#print("Ird_neighbor", Ird_neighbor) ;
				#Ird_neighbor = self.all_Ird_dist[neighborX] ;
				#Ird_itself = self.all_Ird_dist[k] ;
				Ird_itself = self.same_Ird_dist[k] ;
				ratio = Ird_neighbor/Ird_itself ;
				temp_Ird_ratio.append(ratio) ;
			#print("temp_Ird_ratio", temp_Ird_ratio) ;
			#print("sum(temp_Ird_ratio)", sum(temp_Ird_ratio)) ;
			LOF_value = sum(temp_Ird_ratio)/(len(temp_Ird_ratio)) ;
			#print("LOF_value", LOF_value) ;
			self.all_LOF_list.append(LOF_value) ;# different label LOF
class SVM_Outlier:
	train_x_copy= [] ;
	train_y_copy = [];
	record_model = [] ;
	format_x = [] ;
	weight = []
	nr_class = 0 ;
	label = [] ;
	nr_feature = 0 ;
	bias = 0 ;

	value = [] ;

	total_value = [] ;# record all the points total value   Purpose: distinguish whether the point on the right side or not. If total_value>0, right side. otherwise wrong side
	compare_distance = [] ;# absolute distance between weighted line and point

	false_index = [] ;# record index which is not in right label
	false_range = [] ;# record false index's range between the right weighted line

	test_total = 0 ;
	test_accuracy = 0 ;

	def __init__(self, train_x_copy, train_y_copy, record_model):
		self.train_x_copy = train_x_copy ;
		self.train_y_copy = train_y_copy ;
		self.record_model = record_model ;

	def refresh(self):
		self.format_x = [] ;
		self.weight = [] ;
		self.nr_class = 0 ;
		self.nr_feature = 0 ;
		self.bias = 0 ;
		self.label = [] ;
		self.value = [] ;
		self.total_value = [] ;
		self.false_index = [] ;
		self.false_range = [];
		self.compare_distance = [] ;

	def reconstruct_X(self):
		for i in range(len(self.train_x_copy)):
			read_feature = [] ;
			read_value = [] ;
			temp_fragement = [] ;
			match = [] ;
			format_x_temp = [] ;
			temp_fragement_str = "";
			temp_row_last = "" ;
			temp_row = "" ;
			c = 'c' ;
			ptr = 0 ;# read_value pointer

			temp_row = str(self.train_x_copy[i]) ;

			for j in range(len(temp_row)):
				c =  temp_row[j] ;
				if(ord(c)>=44 and ord(c)<=58):
					temp_fragement_str = temp_fragement_str + c ;
		
			temp_fragement = temp_fragement_str.split(',');


			for j in range(len(temp_fragement)):
				temp = "" ;
				temp_array = [] ;
				temp = str(temp_fragement[j]) ;
				temp_array = temp.split(':') ;
				read_feature.append((int)(temp_array[0])) ;

				
				
				try:
					read_value.append((float)(temp_array[1])) ;
				except ValueError:
					read_value.append(-0.00000001) ;
					print("i", i) ;
					print("j", j) ;
					print("y", self.train_y_copy[i]) ;
					print(temp_array[1]) ;


			for j in range(1,self.nr_feature+1,1):#feature start at 1
				if (j in read_feature):
					format_x_temp.append(read_value[ptr]) ;
					ptr = ptr + 1 ;
				else:
					format_x_temp.append(0) ;

			self.format_x.append(format_x_temp) ;

		#print("format_x[1]",format_x[1]) ;
		"""
		weight  = [0.01384472438273333,0.01442689397791132,-0.0007031121332655758,-0.008830199052905132] ;
		for i in range(len(train_x_copy)):
			total_value = bias ;
			for j in range(len(weight)):
				total_value = total_value + weight[j]*format_x[i][j] ;

			if(total_value > 0):
				prediction.append(1) ;
			else:
				prediction.append(0) ;

		test = 0 ;
		for i in range(2000):
			if(prediction[i] == 1):
				test = test + 1 ;

		print("2000 test:",test) ;
		
		for i in range(len(train_y_copy)):
			if(train_y_copy[i] == prediction[i]):
				total = total + 1 ;
		AC = total/(float)(len(train_y_copy)) ;
		print("accuracy", AC) ;
		"""
	def load_model_parameter(self):
		self.nr_class = ((int)(self.record_model[1][1])) ;

		self.label = [] ;

		for i in range(1 ,len(self.record_model[2]), 1):
			self.label.append((int)(self.record_model[2][i])) ;
		self.nr_feature = (int)(self.record_model[3][1]) ;
		bias = 0 ; # (int)(self.record_model[4][1]) ;
		if(self.nr_class == 2):
			count = 1 ;
		else:
			count = self.nr_class ;
		for i in range(count):
			temp_weight = [] ;

			for j in range(6, len(self.record_model), 1):
				temp_weight.append((float)(self.record_model[j][i])) ;

			self.weight.append(temp_weight) ;
	def calculate_value(self):
		for i in range(len(self.train_x_copy)):
			temp_total_value = 0 ;
			temp_weight = [] ;
			sum = 0 ;
			
			if(self.nr_class == 2):
				temp_weight = self.weight[0] ;
			else:
				#print("label") ;
				#print(self.label) ;
				weight_index = self.label.index((int)(self.train_y_copy[i])) ;# choose correct weight

				if(i<=11):
					print("label", i) ;
					print(self.label[weight_index]) ;

				temp_weight = self.weight[weight_index] ;

			for j in range(len(temp_weight)):
				temp_total_value = temp_total_value + temp_weight[j]*self.format_x[i][j] ;# if total

				if(i<=11):
					print("i", self.format_x[i][j]) ;

				sum = sum + temp_weight[j]*temp_weight[j] ;
			
			denominator = math.sqrt(sum) ;
			temp_compare_distance = temp_total_value/denominator ;
			self.total_value.append(temp_total_value) ;
			if(self.nr_class == 2):
				self.compare_distance.append(abs(temp_compare_distance)) ;
			else:
				self.compare_distance.append(temp_compare_distance) ;
			
			
			
			if(self.nr_class == 2):
				####### do prediction to check our theorem
				#######
				if(temp_total_value>0):
					self.value.append(self.label[0]) ;
				else:
					self.value.append(self.label[1]) ;
			"""
			else:
				if(temp_compare_distance>0):
					self.value.append(self.train_y_copy[i]) ;
				else:
					self.false_index.append(i) ;
					#calculate range between weighted line, ranger bigger, chance to be kick off bigger
					print("not in train_y_copy") ;
			#self.value.append(temp_compare_distance) ;
			"""
			
	def prediction_compare(self):
		if(self.nr_class == 2):
			for i in range(len(self.train_x_copy)):
				if(self.train_y_copy[i] == self.value[i]):
					self.test_total = self.test_total + 1 ;
				else:
					self.compare_distance[i] = 0 - (self.compare_distance[i]);




		for i in range(len(self.train_x_copy)):
			if(self.train_y_copy[i] == self.value[i]):
				self.test_total = self.test_total + 1 ;

		self.test_accuracy = self.test_total/(float)(len(self.train_y_copy)) ;
		#print("AC accuracy", self.test_accuracy) ;
		

			

			

	def main_control(self):
		self.refresh() ;
		self.load_model_parameter() ;
		self.reconstruct_X() ;
		self.calculate_value() ;
		if(self.nr_class == 2):
			self.prediction_compare() ;

		





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
		"""
		record_model = [] ;
		with open(pure_data_name+'-temp.train.model') as fp:
			for line in fp:
				temp_str = "" ;
				temp_array = [] ;
				temp_record_model = [] ;
				temp_str = str(line) ;
				temp_str = temp_str.rstrip('\n') ;
				temp_array = temp_str.split(' ') ;

				for j in range(len(temp_array)):
					temp_record_model.append(temp_array[j]) ;

				record_model.append(temp_record_model) ;

		svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, record_model) ;
		svm_outlier.main_control() ;
		print("nr_class", svm_outlier.nr_class) ;
		print("label", svm_outlier.label) ;
		print("nr_feature", svm_outlier.nr_feature) ;
		print("bias", svm_outlier.bias) ;
		"""
		"""
		format_x = [] ;
		nr_class = 2 ;
		label = [1,0] ;
		number_of_feature = 4 ;
		prediction = [];
		bias = 0 ;
		total = 0 ;

		record_model = [] ;
		with open(pure_data_name+'-temp.train.model') as fp:
			for line in fp:
				temp_str = "" ;
				temp_array = [] ;
				temp_record_model = [] ;
				temp_str = str(line) ;
				temp_str = temp_str.rstrip('\n') ;
				temp_array = temp_str.split(' ') ;

				for j in range(len(temp_array)):
					temp_record_model.append(temp_array[j]) ;

				record_model.append(temp_record_model) ;

		print("record_model") ;
		for i in range(20):
			print(i) ;
			print(record_model[i]) ;


				
				
		
		#value = 0.01673985079826928 * x1 + 0.01571213174904187 * x2 - 0.001405405001835395 * x3 - 0.009803531349488146 * x4 - bias ;

		for i in range(len(train_x_copy)):
			read_feature = [] ;
			read_value = [] ;
			temp_fragement = [] ;
			match = [] ;
			format_x_temp = [] ;
			temp_fragement_str = "";
			temp_row_last = "" ;
			temp_row = "" ;
			c = 'c' ;
			ptr = 0 ;# read_value pointer

			temp_row = str(train_x_copy[i]) ;

			for j in range(len(temp_row)):
				c =  temp_row[j] ;
				if(ord(c)>=44 and ord(c)<=58):
					temp_fragement_str = temp_fragement_str + c ;
		
			temp_fragement = temp_fragement_str.split(',');


			for j in range(len(temp_fragement)):
				temp = "" ;
				temp_array = [] ;
				temp = str(temp_fragement[j]) ;
				temp_array = temp.split(':') ;
				read_feature.append((int)(temp_array[0])) ;
				read_value.append((float)(temp_array[1])) ;
	
			for j in range(1,number_of_feature+1,1):#feature start at 1
				if (j in read_feature):
					format_x_temp.append(read_value[ptr]) ;
					ptr = ptr + 1 ;
				else:
					format_x_temp.append(0) ;

			format_x.append(format_x_temp) ;

		print("format_x[1]",format_x[1]) ;

		weight  = [0.01384472438273333,0.01442689397791132,-0.0007031121332655758,-0.008830199052905132] ;
		for i in range(len(train_x_copy)):
			total_value = bias ;
			for j in range(len(weight)):
				total_value = total_value + weight[j]*format_x[i][j] ;

			if(total_value > 0):
				prediction.append(1) ;
			else:
				prediction.append(0) ;

		test = 0 ;
		for i in range(2000):
			if(prediction[i] == 1):
				test = test + 1 ;

		print("2000 test:",test) ;
		
		for i in range(len(train_y_copy)):
			if(train_y_copy[i] == prediction[i]):
				total = total + 1 ;
		AC = total/(float)(len(train_y_copy)) ;
		print("accuracy", AC) ;
		"""
		


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
			cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
			subprocess.call (cmd.split()) ;
			m2 = load_model (data_name+'-temp2.train.model') ;
			# m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
			
			p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
			E_in.append(p_acc[0]/base_acc_in) ;
			p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
			E_out.append(p_acc[0]/base_acc_out) ;
	elif (select_method == 7) :

		for i in range (data_num_size) :
			for j in range (base_size) :
				record_model = [] ;
				with open(data_name+'-temp2.train.model') as fp: # data_name+'-temp2.train.model
					for line in fp:
						temp_str = "" ;
						temp_array = [] ;
						temp_record_model = [] ;
						temp_str = str(line) ;
						temp_str = temp_str.rstrip('\n') ;
						temp_array = temp_str.split(' ') ;

						for j in range(len(temp_array)):
							temp_record_model.append(temp_array[j]) ;
						record_model.append(temp_record_model) ;

				#print("i", i) ;

				svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, record_model) ;
				svm_outlier.main_control() ;
				"""
				print("nr_class", svm_outlier.nr_class) ;
				print("label", svm_outlier.label) ;
				print("nr_feature", svm_outlier.nr_feature) ;
				print("bias", svm_outlier.bias) ;
				"""
				temp_pop_index = svm_outlier.compare_distance.index(min(svm_outlier.compare_distance)) ;# min

				print(temp_pop_index, svm_outlier.compare_distance[temp_pop_index]) ;
				
				# temp_int = len(train_x_copy)-1 ;
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



		"""
		LOF_point = LOF(train_x_copy, train_y_copy, K, MinPts) ; 
		LOF_point.main_control() ; 

		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_pop_index = LOF_point.all_LOF_list.index(max(LOF_point.all_LOF_list)) ;#max
				#temp_pop_index = dst_list.index(max(dst_list)) ;
				LOF_point.all_LOF_list.pop(temp_pop_index) ;
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
		 """
	elif (select_method == 8) :
		"""
		Different_LOF_point = Different_Label_LOF(train_x_copy, train_y_copy, K, MinPts) ; 
		Different_LOF_point.main_control() ; 

		for i in range (data_num_size) :
			for j in range (base_size) :
				
				
				#temp_pop_index = LOF_point.all_LOF_list.index(max(LOF_point.all_LOF_list)) ;#max
				temp_LOF = max(Different_LOF_point.all_LOF_list) ;
				temp_pop_index = Different_LOF_point.all_LOF_list.index(temp_LOF) ;
				#temp_pop_index = Different_LOF_point.all_LOF_list.index(max(Different_LOF_point.all_LOF_list)) ;#max
				#temp_pop_index = dst_list.index(max(dst_list)) ;
				Different_LOF_point.all_LOF_list.pop(temp_pop_index) ;
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
			"""
	
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
print("m",m) ;
print("\n") ;
p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
base_acc_in = p_acc_in[0] ;
p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;
base_acc_out = p_acc_out[0] ;
print("p_acc_out", p_acc_out) ;
E_in_0 = [1.0 for i in range(data_num_size+1)] ;
E_out_0 = [1.0 for i in range(data_num_size+1)] ;

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

select_method = 7 ;
E_in_7 , E_out_7 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('SVM_Outlier') ;
print (E_out_7) ;

"""
select_method = 8 ;
E_in_8 , E_out_8 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Different_LOF') ;
print (E_out_8) ;
"""


# ------------------------------------
"""
cmd = 'rm ' + pure_data_name+'-temp.train' ;
subprocess.call (cmd.split()) ;
cmd = 'rm ' + pure_data_name+'-temp.train.model' ;
subprocess.call (cmd.split()) ;

cmd = 'rm ' + pure_data_name+'-temp2.train' ;
subprocess.call (cmd.split()) ;
cmd = 'rm ' + pure_data_name+'-temp2.train.model' ;
subprocess.call (cmd.split()) ;
"""

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

plt.title(pure_data_name + '_in_' + ('%.3f' % base_acc_in)) ;
plt.plot(query_num, E_in_0, 'k', label='total') ;
plt.plot(query_num, E_in_1, 'bo--', label='random') ;
plt.plot(query_num, E_in_2, 'rv--', label='Euc pair') ;
plt.plot(query_num, E_in_3, 'g^--', label='Euc mean') ;
plt.plot(query_num, E_in_4, 'c*--', label='Man pair') ;
plt.plot(query_num, E_in_5, 'mx--', label='Man mean') ;
plt.plot(query_num, E_in_6, 'yx--', label='Mah mean') ;
plt.plot(query_num, E_in_7, 'rx--', label='SVM_Outlier') ;
#plt.plot(query_num, E_in_8, 'kx--', label='DIF_LOF Man') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_in' + '.png') ;

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

plt.title(pure_data_name + '_out' + ('%.3f' % base_acc_out)) ;
plt.plot(query_num, E_out_0, 'k', label='total') ;
plt.plot(query_num, E_out_1, 'bo--', label='random') ;
plt.plot(query_num, E_out_2, 'rv--', label='Euc pair') ;
plt.plot(query_num, E_out_3, 'g^--', label='Euc mean') ;
plt.plot(query_num, E_out_4, 'c*--', label='Man pair') ;
plt.plot(query_num, E_out_5, 'mx--', label='Man mean') ;
plt.plot(query_num, E_out_6, 'yx--', label='Mah mean') ;
plt.plot(query_num, E_out_7, 'rx--', label='SVM_Outlier') ;
#plt.plot(query_num, E_out_8, 'kx--', label='DIF_LOF Man') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_out' + '.png') ;

plt.cla() ;

# ------------------------------------