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
import statistics ;

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
	weight = [] ;
	wrong_label_index = [] ;
	nr_class = 0 ;
	label = [] ;
	nr_feature = 0 ;
	bias = 0 ;
	nr_wrong_label = 0 ;

	value = [] ;

	#total_value = [] ;# record all the points total value   Purpose: distinguish whether the point on the right side or not. If total_value>0, right side. otherwise wrong side
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
		self.nr_wrong_label = 0 ;# record the number of wrong label node from SVM line
		self.percent_of_wrong_label = 0.0 ;
		self.label = [] ;
		self.value = [] ;
		#self.total_value = [] ;
		self.false_index = [] ;
		self.false_range = [];
		self.compare_distance = [] ;
		self.wrong_label_index = [] ;

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
					"""
					print("i", i) ;
					print("j", j) ;
					print("y", self.train_y_copy[i]) ;
					print(temp_array[1]) ;
					"""


			for j in range(1,self.nr_feature+1,1):#feature start at 1
				if (j in read_feature):
					format_x_temp.append(read_value[ptr]) ;
					ptr = ptr + 1 ;

				else:
					format_x_temp.append(0) ;

			self.format_x.append(format_x_temp) ;

		#print("format_x[1]",format_x[1]) ;
	def load_model_parameter(self):
		self.nr_class = ((int)(self.record_model[1][1])) ;

		self.label = [] ;

		for i in range(1 ,len(self.record_model[2]), 1):
			self.label.append((int)(self.record_model[2][i])) ;
		self.nr_feature = (int)(self.record_model[3][1]) ;
		# (int)(self.record_model[4][1]) ;
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
			temp_total_value = self.bias ;
			temp_weight = [] ;
			sum = 0 ;
			
			if(self.nr_class == 2):
				temp_weight = self.weight[0] ;
			else:
				#print("label") ;
				#print(self.label) ;
				weight_index = self.label.index((int)(self.train_y_copy[i])) ;# choose correct weight

				#if(i<=11):
					#print("label", i) ;
					#print(self.label[weight_index]) ;

				temp_weight = self.weight[weight_index] ;

			for j in range(len(temp_weight)):
				temp_total_value = temp_total_value + temp_weight[j]*self.format_x[i][j] ;# if total

				#if(i<=11):
					#print("i", self.format_x[i][j]) ;

				sum = sum + temp_weight[j]*temp_weight[j] ;
			
			denominator = math.sqrt(sum) ;
			temp_compare_distance = temp_total_value/denominator ;
			#self.total_value.append(temp_total_value) ;
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
					self.nr_wrong_label = self.nr_wrong_label + 1 ;
					self.wrong_label_index.append(i) ;
		else:
			for i in range(len(self.compare_distance)):
				if(self.compare_distance[i]<0):
					self.nr_wrong_label = self.nr_wrong_label + 1 ;
					self.wrong_label_index.append(i) ;

		self.percent_of_wrong_label = self.nr_wrong_label/(float)(len(self.train_x_copy)) ;
		self.percent_of_wrong_label = self.percent_of_wrong_label * 100 ;



		if(self.nr_class == 2):
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
		self.prediction_compare() ;
class SVM_Outlier_Angle:# number of class = 2
	train_x_copy= [] ;
	train_y_copy = [];
	record_model = [] ;
	format_x = [] ;
	weight = []
	nr_class = 0 ;
	label = [] ;
	nr_feature = 0 ;
	bias = 0 ;
	nr_wrong_label = 0 ;

	right_label_index = [] ;
	right_label_X = [] ;
	right_label_Y = [] ;
	wrong_label_index = [] ;
	wrong_label_X = [] ;
	wrong_label_Y = [] ;

	vector_cos = [] ;
	variance_cos = [] ;# method key

	value = [] ;

	#total_value = [] ;# record all the points total value   Purpose: distinguish whether the point on the right side or not. If total_value>0, right side. otherwise wrong side
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
		self.nr_wrong_label = 0 ;# record the number of wrong label node from SVM line
		self.percent_of_wrong_label = 0.0 ;
		self.label = [] ;
		self.value = [] ;
		#self.total_value = [] ;
		self.false_index = [] ;
		self.false_range = [];
		self.compare_distance = [] ;

		self.right_label_index = [] ;
		self.right_label_X = [] ;
		self.right_label_Y = [] ;
		self.wrong_label_index = [] ;
		self.wrong_label_X = [] ;
		self. wrong_label_Y = [] ;

		self.vector_cos = [] ;
		self.variance_cos = [] ;# method key

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
					"""
					print("i", i) ;
					print("j", j) ;
					print("y", self.train_y_copy[i]) ;
					print(temp_array[1]) ;
					"""


			for j in range(1,self.nr_feature+1,1):#feature start at 1
				if (j in read_feature):
					format_x_temp.append(read_value[ptr]) ;
					ptr = ptr + 1 ;

				else:
					format_x_temp.append(0) ;

			self.format_x.append(format_x_temp) ;

	def load_model_parameter(self):
		self.nr_class = ((int)(self.record_model[1][1])) ;

		self.label = [] ;

		for i in range(1 ,len(self.record_model[2]), 1):
			self.label.append((int)(self.record_model[2][i])) ;
		self.nr_feature = (int)(self.record_model[3][1]) ;
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
			temp_total_value = self.bias ;
			temp_weight = [] ;
			sum = 0 ;
			
			if(self.nr_class == 2):
				temp_weight = self.weight[0] ;
			else:
				#print("label") ;
				#print(self.label) ;
				weight_index = self.label.index((int)(self.train_y_copy[i])) ;# choose correct weight

				#if(i<=11):
					#print("label", i) ;
					#print(self.label[weight_index]) ;

				temp_weight = self.weight[weight_index] ;

			for j in range(len(temp_weight)):
				temp_total_value = temp_total_value + temp_weight[j]*self.format_x[i][j] ;# if total

				#if(i<=11):
					#print("i", self.format_x[i][j]) ;

				sum = sum + temp_weight[j]*temp_weight[j] ;
			
			denominator = math.sqrt(sum) ;
			temp_compare_distance = temp_total_value/denominator ;
			#self.total_value.append(temp_total_value) ;
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
					self.right_label_index.append(i) ;
					self.right_label_X.append(self.format_x[i]) ;
					self.right_label_Y.append(self.train_y_copy[i]) ;
					self.test_total = self.test_total + 1 ;
				else:
					self.compare_distance[i] = 0 - (self.compare_distance[i]);
					self.wrong_label_index.append(i) ;
					self.wrong_label_X.append(self.format_x[i]) ;
					self.wrong_label_Y.append(self.train_y_copy[i]) ;
					self.nr_wrong_label = self.nr_wrong_label + 1 ;
		else:
			for i in range(len(self.compare_distance)):
				if(self.compare_distance[i]<0):
					self.nr_wrong_label = self.nr_wrong_label + 1 ;

		self.percent_of_wrong_label = self.nr_wrong_label/(float)(len(self.train_x_copy)) ;
		self.percent_of_wrong_label = self.percent_of_wrong_label * 100 ;



		if(self.nr_class == 2):
			for i in range(len(self.train_x_copy)):
				if(self.train_y_copy[i] == self.value[i]):
					self.test_total = self.test_total + 1 ;

			self.test_accuracy = self.test_total/(float)(len(self.train_y_copy)) ;

	def prediction_angle(self):
		for i in range(len(self.wrong_label_X)):
			#print("i:", i) ;
			temp_vector_cos = [] ;
			vector_1 = [] ;
			vector_2 = [] ;
			wrong_1 = self.wrong_label_X[i] ;
			wrong_1_Y = self.wrong_label_Y[i] ;
			for j in range(len(self.right_label_index)):
				right_1 = self.right_label_X[j] ;
				right_1_Y = self.right_label_Y[j] ;
				if(wrong_1_Y == right_1_Y):
					break ;
			vector_1 = self.vector_minus(right_1, wrong_1) ;
			for k in range( j+1, len(self.right_label_index), 1):
				right_2 = self.right_label_X[k] ;
				right_2_Y = self.right_label_Y[k] ;
				if(wrong_1_Y == right_2_Y):
					vector_2 = self.vector_minus(right_2, wrong_1) ;
					inner_product = self.vector_inner_product(vector_1, vector_2) ;

					vector_1_pow = self.vector_pow(vector_1) ;
					vector_2_pow = self.vector_pow(vector_2) ;

					cos = inner_product/(vector_1_pow*vector_2_pow) ;
					temp_vector_cos.append(cos) ;
			self.vector_cos.append(temp_vector_cos) ;

		"""
		print("right_label_index", len(self.right_label_index)) ;
		for i in range(len(self.wrong_label_X)):
			print("i:", i) ;
			temp_vector_cos = [] ;
			vector_1 = [] ;
			wrong_1 = self.wrong_label_X[i] ;
			wrong_1_Y = self.wrong_label_Y[i] ;
			for j in range(len(self.right_label_index)):
				right_1 = self.right_label_X[j] ;
				right_1_Y = self.right_label_Y[j] ;
				if(wrong_1_Y == right_1_Y):
					vector_1 = self.vector_minus(right_1, wrong_1) ;
					vector_2 = [] ;
					count = 0 ;
					while count<=10 :
						k = randint(0, len(self.right_label_index)-1) ;
						right_2 = self.right_label_X[k] ;
						right_2_Y = self.right_label_Y[k] ;
						if(wrong_1_Y == right_2_Y):
							count = count + 1 ;
							vector_2 = self.vector_minus(right_2, wrong_1) ;

							inner_product = self.vector_inner_product(vector_1, vector_2) ;

							vector_1_pow = self.vector_pow(vector_1) ;
							vector_2_pow = self.vector_pow(vector_2) ;

							cos = inner_product/(vector_1_pow*vector_2_pow) ;
							temp_vector_cos.append(cos) ;
		"""
					
	def angle_variance(self):
		for i in range(len(self.vector_cos)):
			#print(self.vector_cos[i]) ;
			var = statistics.variance(self.vector_cos[i]) ;
			self.variance_cos.append(var) ;
	def vector_minus(self, vec_1, vec_2):
		vector = [] ;
		temp = 0.0 ;
		for i in range(len(vec_1)):
			temp = vec_1[i] - vec_2[i] ;
			vector.append(temp) ;
		return vector ;
	def vector_plus(self, vec_1, vec_2):
		vector = [] ;
		temp = 0.0 ;
		for i in range(len(vec_1)):
			temp = vec_1 + vec_2 ;
			vector.append(temp) ;
		return vector ;
	def vector_inner_product(self, vec_1, vec_2):
		inner_product = 0.0 ;
		for i in range(len(vec_1)):
			inner_product = inner_product + vec_1[i] * vec_2[i] ;
		return inner_product ;
	def vector_pow(self, vec_1):
		temp = 0.0 ;
		for i in range(len(vec_1)):
			temp = temp + vec_1[i] * vec_1[i] ;
		return temp ;
	def main_control(self):
		self.refresh() ;
		self.load_model_parameter() ;
		self.reconstruct_X() ;
		self.calculate_value() ;
		if(self.nr_class == 2):
			self.prediction_compare() ;
			self.prediction_angle() ;
			self.angle_variance() ;
class General:
	train_x_copy = [] ;
	record_model = [] ;
	format_x = [] ;
	weight = []
	nr_class = 0 ;
	label = [] ;
	nr_feature = 0 ;
	def __init__(self, train_x_copy, record_model):
		self.train_x_copy = copy.deepcopy(train_x_copy) ;
		self.record_model = copy.deepcopy(record_model) ;
	def load_model_parameter(self):
		self.nr_class = ((int)(self.record_model[1][1])) ;

		self.label = [] ;

		for i in range(1 ,len(self.record_model[2]), 1):
			self.label.append((int)(self.record_model[2][i])) ;
		self.nr_feature = (int)(self.record_model[3][1]) ;
		if(self.nr_class == 2):
			count = 1 ;
		else:
			count = self.nr_class ;
		for i in range(count):
			temp_weight = [] ;

			for j in range(6, len(self.record_model), 1):
				temp_weight.append((float)(self.record_model[j][i])) ;

			self.weight.append(temp_weight) ;

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
					"""
					print("i", i) ;
					print("j", j) ;
					print("y", self.train_y_copy[i]) ;
					print(temp_array[1]) ;
					"""


			for j in range(1,self.nr_feature+1,1):#feature start at 1
				if (j in read_feature):
					format_x_temp.append(read_value[ptr]) ;
					ptr = ptr + 1 ;

				else:
					format_x_temp.append(0) ;

			self.format_x.append(format_x_temp) ;
	def refresh(self):
		#train_x_copy = [] ;
		#record_model = [] ;
		self.format_x = [] ;
		self.weight = []
		self.nr_class = 0 ;
		self.label = [] ;
		self.nr_feature = 0 ;
	def main_control(self):
		self.refresh() ;
		self.load_model_parameter() ;
		self.reconstruct_X() ;

		
class ROF_Method:
	train_x_copy= [] ;
	train_y_copy = [];
	CR_List = [] ; # record every round of clusterSize and ROF value
	nr_each_label = [] ;
	resolution_max = 0.0 ;
	resolution_min = 0.0 ;
	current_r = 0.0 ;
	delta_r_percent = 0.1 ;# 4% - 10% #a1a 0.3
	nr_topN = 5 ;
	All_Is_Cluster = False ;
	number_of_point = 0 ;
	number_of_unmerged_point = 1000 ;

	def __init__(self, train_x_copy, train_y_copy) :
		self.train_x_copy = [] ;
		self.train_y_copy = [] ;
		self.train_x_copy = copy.deepcopy(train_x_copy) ;
		self.train_y_copy = copy.deepcopy(train_y_copy) ;

		#self.train_x_copy = train_x_copy ;
		#self.train_y_copy = train_y_copy ;
		print("IN_nr_train_x_copy", len(self.train_x_copy)) ;
		print("IN_nr_train_y_copy", len(self.train_y_copy)) ;
		self.number_of_point = len(self.train_x_copy) ;

	def refresh(self):
		self.CR_List = [] ; # record every round of clusterSize and ROF value
		self.nr_each_label = [] ; # record  nuber of each label
		self.resolution_max = 0.0 ;
		self.resolution_min = 0.0 ;
		self.current_r = 0.0 ;
		self.delta_r_percent = 100 ;# 4% - 10%  a1a:0.25
		self.nr_topN = 50 ;
		self.All_Is_Cluster = False ;
		self.number_of_unmerged_point = 1000 ;
		
		for i in range(self.number_of_point):
			self.nr_each_label.append(1) ;

	def calculate_resolution_range(self):
		#max_min_neiDist = 0.0 ;
		#min_min_neiDist = 0.0 ;
		min_neiDist = [] ;
		for i in range(len(self.train_x_copy)):# 1605
			minimum = 1000000000.0 ;
			for j in range(len(self.train_x_copy)):
				if(self.train_y_copy[i] == self.train_y_copy[j] and i != j):
					temp = self.euc_dist(self.train_x_copy[i], self.train_x_copy[j]) ;
					if(temp<minimum):
						minimum = temp ;
				else:
					continue ;
			min_neiDist.append(minimum) ;
		max_min_neiDist = max(min_neiDist) ;
		min_min_neiDist = min(min_neiDist) ;
		while(min_min_neiDist<=0):
			pop_index = min_neiDist.index(min_min_neiDist) ;
			min_neiDist.pop(pop_index) ;
			min_min_neiDist = min(min_neiDist) ;
		self.resolution_min = 1/max_min_neiDist ;
		self.resolution_max = 1/min_min_neiDist ;
		"""
		if(self.resolution_min * 10 < self.resolution_max):
			self.resolution_min = self.resolution_max/4.0 ;
		"""
	class Cluster_ROF_Data: # record one round
		ClusterSize = [] ;
		ROF = [] ;
		label = [] ;
		def __init__(self) :
			print("OO") ;
		def start(self, number_of_point):
			self.ClusterSize = [] ;
			self.ROF = [] ;
			self.label = [] ;
			for i in range(number_of_point):
				self.ClusterSize.append(1) ;
				self.ROF.append(0) ;
				self.label.append(i) ;

	def RB_MINE(self):
		Ind_CR_List = self.Cluster_ROF_Data() ;
		Ind_CR_List.start(len(self.train_x_copy)) ;
		self.CR_List.append(Ind_CR_List) ;
		self.calculate_resolution_range() ;# calculate max resolution and min resolution
		current_r = self.resolution_max ;

		self.All_Is_Cluster = True ;
		round = 1 ;
		while (self.All_Is_Cluster == True) and (self.number_of_unmerged_point > self.nr_topN):
		#while(current_r>=self.resolution_min):
			self.All_Is_Cluster = False ;
			current_r = current_r - (current_r - self.resolution_min) * self.delta_r_percent ;
			self.RB_CLUSTER( round, current_r) ;
			
			#print("round",round) ;
			round = round + 1 ;
	def RB_CLUSTER(self, round, current_r):# start from round 1
		self.number_of_unmerged_point = 0 ;
		Ind_CR_List = self.Cluster_ROF_Data() ;
		for i in range(self.number_of_point):
			Ind_CR_List.ClusterSize = copy.deepcopy(self.CR_List[round-1].ClusterSize) ;
			Ind_CR_List.label = copy.deepcopy(self.CR_List[round-1].label) ;
			Ind_CR_List.ROF = copy.deepcopy(self.CR_List[round-1].ROF) ;
		self.CR_List.append(Ind_CR_List) ;

		for i in range(len(self.train_x_copy)):# reset coordinate by resolution r
			for j in range(len(self.train_x_copy[i])):
				self.train_x_copy[i][j] = self.train_x_copy[i][j] * current_r ;

		for i in range(self.number_of_point):# update Label and ClusterSize
			Merged_point = False ;
			for j in range(i, self.number_of_point, 1):
				if(self.train_y_copy[i] == self.train_y_copy[j]  and self.CR_List[round].label[i] != self.CR_List[round].label[j]  and i != j ):
					temp_dist = self.euc_dist(self.train_x_copy[i], self.train_x_copy[j]) ;
					if(temp_dist<=1):
						self.All_Is_Cluster = True ;
						Merged_point = True ;
						if(self.CR_List[round].label[i]<self.CR_List[round].label[j]):# update label
							self.CR_List[round].label[j] = self.CR_List[round].label[i] ;
							self.nr_each_label[i] = self.nr_each_label[i] + 1 ;
						else:
							self.CR_List[round].label[i] = self.CR_List[round].label[j] ;
							self.nr_each_label[j] = self.nr_each_label[j] + 1 ;
						#self.CR_List[round].ClusterSize[i] = self.CR_List[round].ClusterSize[i] + 1 ;
						#self.CR_List[round].ClusterSize[j] = self.CR_List[round].ClusterSize[j] + 1 ;
			if(Merged_point == False): # record number_of_unmerged_point 
				self.number_of_unmerged_point = self.number_of_unmerged_point + 1 ;

		# synchronized each point's ClusterSize
		for i in range(self.number_of_point):
			label = self.CR_List[round].label[i] ;
			self.CR_List[round].ClusterSize[i] = self.nr_each_label[label] ;

		# update ROF
		for i in range(self.number_of_point):
			self.CR_List[round].ROF[i] = self.CR_List[round-1].ROF[i] + ((self.CR_List[round-1].ClusterSize[i] - 1)/(float)(self.CR_List[round].ClusterSize[i])) ;

	def main_control(self):
		self.refresh() ;
		self.RB_MINE() ;
		self.train_x_copy = [] ;
		self.train_y_copy = [] ;

					
	def euc_dist(self, a, b):
		temp = 0.0 ;
		for i in range(len(a)):
			temp = temp + ((a[i] - b[i]) * (a[i] - b[i])) ;
		temp = math.pow(temp, 0.5) ;
		return temp ;
		





		





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
	"""
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

	gen =  General(train_x_copy, record_model) ;
	gen.load_model_parameter() ;
	gen.reconstruct_X() ;


	rof = ROF_Method(gen.format_x, train_y_copy) ;
	rof.main_control() ;
	for i in range(len(rof.CR_List)):
		print("\n") ;
		print("i", i) ;
		print("\n") ;
		for j in range(10):
			print("label", rof.CR_List[i].label[j]) ;
			print("ClusterSize", rof.CR_List[i].ClusterSize[j]) ;
			print("ROF", rof.CR_List[i].ROF[j]) ;
	"""

	
	if (select_method == 1) :
		# hybrid

		for i in range (data_num_size) :
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
			for j in range (base_size) :
				temp_int = randint (0,len(train_x_copy)-1) ;
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



		"""
		for i in range (data_num_size) :
			for j in range (base_size) :
				temp_int = randint (0,len(train_x_copy)-1) ;
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
		"""

	elif (select_method == 2) :

		for i in range (data_num_size) :
			"""
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

			svm_outlier_angle = SVM_Outlier_Angle(train_x_copy, train_y_copy, record_model) ;
			svm_outlier_angle.main_control() ;
			"""
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
			svm_outlier_angle = SVM_Outlier_Angle(train_x_copy, train_y_copy, record_model) ;
			svm_outlier_angle.main_control() ;
		
			for j in range (base_size) :
	
				#svm_outlier_angle = SVM_Outlier_Angle(train_x_copy, train_y_copy, record_model) ;
				#svm_outlier_angle.main_control() ;
				if(len(svm_outlier_angle.variance_cos)>0):
					temp = svm_outlier_angle.variance_cos.index(min(svm_outlier_angle.variance_cos)) ;
					temp_pop_index = svm_outlier_angle.wrong_label_index[temp] ;
					
					svm_outlier_angle.variance_cos.pop(temp) ;

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
			"""
	
	elif (select_method == 3) :
		for i in range (data_num_size) :
			"""
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
			svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, record_model) ;
			svm_outlier.main_control() ;

			for j in range (base_size) :
				

				#print("i", i) ;

				#svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, record_model) ;
				#svm_outlier.main_control() ;
				
				temp_pop_index = svm_outlier.compare_distance.index(min(svm_outlier.compare_distance)) ;# min

				svm_outlier.compare_distance.pop(temp_pop_index) ;
			
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
			"""
	elif (select_method == 4) :# data svmguide1 resolution_max = 4.25 , resolution_min =0.015 ; a1a resolution_max = 1 , resolution_min = 0.28~0.35
		for i in range (data_num_size) :
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
			"""
			gen =  General(train_x_copy, record_model) ;
			gen.main_control() ;
			#print("OUT_train_x,_copy" , len(gen.format_x)) ;
			#print("OUT_train_y_copy" , len(train_y_copy)) ;
			rof = ROF_Method(gen.format_x, train_y_copy) ;
			rof.main_control() ;
			"""
			svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, record_model) ;
			svm_outlier.main_control() ;

			gen =  General(train_x_copy, record_model) ;
			gen.main_control() ;
			print("OUT_train_x,_copy" , len(gen.format_x)) ;
			print("OUT_train_y_copy" , len(train_y_copy)) ;

			rof = ROF_Method(gen.format_x, train_y_copy) ;
			rof.main_control() ;
			for j in range (base_size) :
				"""
				while(True):
					ROF_min = min(rof.CR_List[len(rof.CR_List)-1].ROF) ;
					temp_int = rof.CR_List[len(rof.CR_List)-1].ROF.index(ROF_min) ;
					rof.CR_List[len(rof.CR_List)-1].ClusterSize.pop(temp_int) ;
					rof.CR_List[len(rof.CR_List)-1].label.pop(temp_int) ;
					rof.CR_List[len(rof.CR_List)-1].ROF.pop(temp_int) ;
					if(temp_int in svm_outlier.wrong_label_index):
						break ;
				"""
				ROF_min = min(rof.CR_List[len(rof.CR_List)-1].ROF) ;
				temp_int = rof.CR_List[len(rof.CR_List)-1].ROF.index(ROF_min) ;
				rof.CR_List[len(rof.CR_List)-1].ClusterSize.pop(temp_int) ;
				rof.CR_List[len(rof.CR_List)-1].label.pop(temp_int) ;
				rof.CR_List[len(rof.CR_List)-1].ROF.pop(temp_int) ;
				print("ROF_min", ROF_min) ;
				
				print("resolution_min", rof.resolution_min) ;
				print("resolution_max", rof.resolution_max) ;
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
		"""
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
			"""
	
	elif (select_method == 5) :

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
			"""
	
	elif (select_method == 6) :
		"""
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
			"""
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

				#print(temp_pop_index, svm_outlier.compare_distance[temp_pop_index]) ;
				
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
def percentage_wrong_label(data_name  , data_num_size ,  train_x , train_y) :
	train_x_copy = copy.deepcopy(train_x) ;
	train_y_copy = copy.deepcopy(train_y) ;
	
	write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
	cmd = '../../liblinear-incdec-2.01/train -s 2 -q ' + data_name+'-temp2.train' ;
	subprocess.call (cmd.split()) ;
	m = load_model (data_name+'-temp2.train.model') ;	
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

	svm_outlier_run = SVM_Outlier(train_x_copy, train_y_copy, record_model) ;
	svm_outlier_run.main_control() ;
	return svm_outlier_run.percent_of_wrong_label ;

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

per_wrong_label = percentage_wrong_label(pure_data_name, data_num_size,train_x , train_y) ;

select_method = 1 ;
E_in_1 , E_out_1= run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('random') ;
print (E_out_1) ;


select_method = 2 ;
E_in_2 , E_out_2 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('svm_outlier_angle') ;
print (E_out_2) ;

select_method = 3 ;
E_in_3 , E_out_3 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('svm_outlier_distance') ;
print (E_out_3) ;


select_method = 4 ;
E_in_4 , E_out_4 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('ROF') ;
print (E_out_4) ;

select_method = 5 ;
E_in_5 , E_out_5 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('LOF') ; # Manhattan distance mean
print (E_out_5) ;
"""
select_method = 6 ;
E_in_6 , E_out_6 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Mahalanobis distance mean') ;
print (E_out_6) ;


select_method = 7 ;
E_in_7 , E_out_7 = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('SVM_Outlier') ;
print (E_out_7) ;


select_method = 8 ;
E_in_8 , E_out_8, per_wrong_label = run (pure_data_name , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
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

plt.title(pure_data_name + '_in_' + ('%.3f' % base_acc_in) + 'wrong_pr' +  ('%.3f' % per_wrong_label)) ;
plt.plot(query_num, E_in_0, 'k', label='total') ;
plt.plot(query_num, E_in_1, 'bo--', label='random') ;
plt.plot(query_num, E_in_2, 'rv--', label='SVM_Angle') ;
plt.plot(query_num, E_in_3, 'g^--', label='SVM_Dist') ;
plt.plot(query_num, E_in_4, 'c*--', label='ROF_NO_SVM') ;
plt.plot(query_num, E_in_5, 'mx--', label='LOF') ; #Man mean
#plt.plot(query_num, E_in_6, 'yx--', label='Mah mean') ;
#plt.plot(query_num, E_in_7, 'rx--', label='SVM_Outlier') ;
#plt.plot(query_num, E_in_8, 'kx--', label='DIF_LOF Man') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_in_ROF_NO_SVM' + '.png') ;

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

plt.title(pure_data_name + '_out' + ('%.3f' % base_acc_out) + 'wrong_pr' +  ('%.3f' % per_wrong_label)) ;
plt.plot(query_num, E_out_0, 'k', label='total') ;
plt.plot(query_num, E_out_1, 'bo--', label='random') ;
plt.plot(query_num, E_out_2, 'rv--', label='SVM_Angle') ;
plt.plot(query_num, E_out_3, 'g^--', label='SVM_Dist') ;
plt.plot(query_num, E_out_4, 'c*--', label='ROF_NO_SVM') ;
plt.plot(query_num, E_out_5, 'mx--', label='LOF') ;# Man mean
#plt.plot(query_num, E_out_6, 'yx--', label='Mah mean') ;
#plt.plot(query_num, E_out_7, 'rx--', label='SVM_Outlier') ;
#plt.plot(query_num, E_out_8, 'kx--', label='DIF_LOF Man') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_out_ROF_NO_SVM' + '.png') ;

plt.cla() ;

# ------------------------------------