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
import pickle ;
import random ;

from itertools import permutations ;
from sklearn import svm ;
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
    record_model = [] ;
    label = [] ;
    label_lof = [] ;

    def __init__(self, train_x_copy, train_y_copy, record_model, parameter_k , MinPts):
        self.train_x_copy = train_x_copy ; 
        self.train_y_copy = train_y_copy ;
        self.record_model = record_model ;
        self.parameter_k = parameter_k ;
        self.MinPts = MinPts ; 
    def main_control(self):
        self.load_model_parameter() ;

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
        self.Average_LOF() ;

    def load_model_parameter(self):
        self.nr_class = ((int)(self.record_model[1][1])) ;

        self.label = [] ;

        for i in range(1 ,len(self.record_model[2]), 1):
            self.label.append((int)(self.record_model[2][i])) ;
        
        for i in range(len(self.label)):
            temp = [] ;
            self.label_lof.append(temp) ;
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
        self.all_LOF_list = [] ;
        for k in range(len(self.train_x_copy)) :
            temp_Ird_ratio = [] ;
            for i in range(self.MinPts):
                neighborX = self.all_neighborX_list[k][i] ;
                Ird_neighbor = self.all_Ird_dist[neighborX] ;
                Ird_itself = self.all_Ird_dist[k] ;
                ratio = Ird_neighbor/Ird_itself ;
                temp_Ird_ratio.append(ratio) ;
            LOF_value = sum(temp_Ird_ratio)/(float)(len(temp_Ird_ratio)) ;

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!! consider local LOF value

            label_y = self.train_y_copy[k] ;
            self.label_lof[self.label.index(label_y)].append(LOF_value) ;
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.all_LOF_list.append(LOF_value) ;# same label LOF
    def Average_LOF(self):
        label_average_list = [] ;
        label_average= 0.0 ;
        if(self.nr_class == 2):
            for i in range(len(self.label)):
                label_average_list.append(np.mean(self.label_lof[i])) ;
            for i in range(len(self.all_LOF_list)):
                label_y = self.train_y_copy[i] ;
                label_average = label_average_list[self.label.index(label_y)] ;
                self.all_LOF_list[i] = self.all_LOF_list[i]/label_average ;
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
    label_dist = [] ;
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
        self.label_dist = [] ;

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
        for i in range(len(self.label)):
            temp = [] ;
            self.label_dist.append(temp) ;
    def calculate_value(self):
        for i in range(len(self.train_x_copy)):
            temp_total_value = self.bias ;
            temp_weight = [] ;
            sum = 0 ;
            
            if(self.nr_class == 2):
                temp_weight = self.weight[0] ;
            else:
                weight_index = self.label.index((int)(self.train_y_copy[i])) ;# choose correct weight
                temp_weight = self.weight[weight_index] ;

            for j in range(len(temp_weight)):
                temp_total_value = temp_total_value + temp_weight[j]*self.format_x[i][j] ;# if total
                sum = sum + temp_weight[j]*temp_weight[j] ;
            denominator = math.sqrt(sum) ;
            temp_compare_distance = temp_total_value/denominator ;
            #self.total_value.append(temp_total_value) ;
            if(self.nr_class == 2):
                label_y = self.train_y_copy[i] ;
                self.label_dist[self.label.index(label_y)].append(abs(temp_compare_distance)) ;
                """
                if(self.train_y_copy[i] == self.label[0]):
                    self.label_dist[0].append(abs(temp_compare_distance)) ;
                else:
                    self.label_dist[1].append(abs(temp_compare_distance)) ;
                """
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
    def distance_average(self):
        label_average_list = [] ;
        label_average= 0.0 ;
        if(self.nr_class == 2):
            for i in range(len(self.label)):
                label_average_list.append(np.mean(self.label_dist[i])) ;
            for i in range(len(self.compare_distance)):
                label_y = self.train_y_copy[i] ;
                label_average = label_average_list[self.label.index(label_y)] ;
                self.compare_distance[i] = self.compare_distance[i]/label_average ;

            

    def main_control(self):
        self.refresh() ;
        self.load_model_parameter() ;
        self.reconstruct_X() ;
        self.calculate_value() ;
        self.distance_average() ;
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

    def __init__(self, train_x_copy, format_x, train_y_copy, record_model):
        self.train_x_copy = train_x_copy ;
        self.format_x = format_x ;
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
        """
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
        
        #print("right_label_index", len(self.right_label_index)) ;
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
            self.vector_cos.append(temp_vector_cos) ;#HAVE PROBLEM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! k nearest neighbor

        
                    
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
    delta_r_percent = 0.04;# 4% - 10% 
    nr_topN = 0 ;
    All_Is_Cluster = False ;
    number_of_point = 0 ;
    number_of_unmerged_point = 1000 ;

    def __init__(self, train_x_copy, train_y_copy, top_n) :
        self.train_x_copy = [] ;
        self.train_y_copy = [] ;
        self.train_x_copy = copy.deepcopy(train_x_copy) ;
        self.train_y_copy = copy.deepcopy(train_y_copy) ;
        self.nr_topN = top_n ;
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
        self.delta_r_percent = 0.1 ;# 4% - 10%  a1a:0.25
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
        #print("resolution_max", self.resolution_max) ;
        #print("resolution_min", self.resolution_min) ;
        self.current_r = self.resolution_max ;
        #print("before while current_r", self.current_r) ;
        self.All_Is_Cluster = True ;
        round = 1 ;
        while  (self.number_of_unmerged_point > self.nr_topN): # (self.All_Is_Cluster == True) and (self.number_of_unmerged_point > self.nr_topN)
        #while(current_r>=self.resolution_min):
            self.All_Is_Cluster = False ;
            self.current_r = self.current_r - (self.resolution_max- self.resolution_min) * self.delta_r_percent ; # current_r = current_r - (current_r - self.resolution_min) * self.delta_r_percent   # self.resolution_max
            #print("\n") ;
            #print("round",round) ;
            #print("current_r", self.current_r) ;
            self.RB_CLUSTER( round, self.current_r) ;
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
            #Merged_point = False ;
            for j in range(i, self.number_of_point, 1):
                if(self.train_y_copy[i] == self.train_y_copy[j]  and self.CR_List[round].label[i] != self.CR_List[round].label[j]  and i != j ):
                    temp_dist = self.euc_dist(self.train_x_copy[i], self.train_x_copy[j]) ;
                    if(temp_dist<=1):
                        self.All_Is_Cluster = True ;
                        #Merged_point = True ;
                        if(self.CR_List[round].label[i]<self.CR_List[round].label[j]):# update label
                            self.CR_List[round].label[j] = self.CR_List[round].label[i] ;
                            self.nr_each_label[i] = self.nr_each_label[i] + 1 ;
                        else:
                            self.CR_List[round].label[i] = self.CR_List[round].label[j] ;
                            self.nr_each_label[j] = self.nr_each_label[j] + 1 ;
                        #self.CR_List[round].ClusterSize[i] = self.CR_List[round].ClusterSize[i] + 1 ;
                        #self.CR_List[round].ClusterSize[j] = self.CR_List[round].ClusterSize[j] + 1 ;
            """
            if(Merged_point == False): # record number_of_unmerged_point 
                self.number_of_unmerged_point = self.number_of_unmerged_point + 1 ;
            """

        # synchronized each point's ClusterSize
        for i in range(self.number_of_point):
            label = self.CR_List[round].label[i] ;
            self.CR_List[round].ClusterSize[i] = self.nr_each_label[label] ;
        # count unmerged points
        for i in range(self.number_of_point):
            temp_size = self.CR_List[round].ClusterSize[i] ;
            if(temp_size == 1):
                self.number_of_unmerged_point = self.number_of_unmerged_point + 1 ;
        print("number_of_unmerged_point", self.number_of_unmerged_point) ;
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
class Angle_Based_FastABOD:
    train_x_copy= [] ;
    train_y_copy = [];
    format_x = [] ;
    data_length = 0 ;
    parameter_k = 0 ;
    nearest_neighbor_list = [] ; # for same label
    error_1 = 0 ;
    error_2 = 0 ;
    error_12 = 0 ;
    #vector_cos = [] ;
    variance_cos = [] ;# method key

    def __init__(self, train_x_copy, format_x, train_y_copy, parameter_k):
        self.train_x_copy = train_x_copy ;
        self.data_length = len(train_x_copy) ;
        self.format_x = format_x ;
        self.train_y_copy = train_y_copy ;
        self.parameter_k = parameter_k ;
    def refresh(self):
        self.train_x_copy= [] ;
        self.train_y_copy = [];
        self.format_x = [] ;
        self.data_length = 0 ;
        self.parameter_k = 0 ;
        self.nearest_neighbor_list = [] ; # for same label
        self.error_1 = 0 ;
        self.error_2 = 0 ;
        self.error_12 = 0 ;
    def main_control(self):
        self.find_nearest_neighbor() ;
        self.prediction_angle() ;
    def find_nearest_neighbor(self): # need to count every pair of point's distance
        self.nearest_neighbor_list = [] ;
        for i in range(self.data_length):
            temp_neighbor = self.neighbor() ;
            temp_neighbor.refresh() ;
            for j in range(self.data_length):
                if(self.train_y_copy[i] == self.train_y_copy[j]): # add itself to the neighbor list
                    temp = [self.train_x_copy[i] , self.train_x_copy[j]] ;
                    words = sorted(list(reduce(set.union, map(set, temp)))) ;
                    feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
                    dist = distance.euclidean(feats[0],feats[1]) ; #dist = distance.cityblock(feats[0],feats[1]) ;
                    if(len(temp_neighbor.dist)<self.parameter_k):
                        temp_neighbor.dist.append(dist) ;
                        temp_neighbor.index.append(j) ;
                    else:
                        if(dist<max(temp_neighbor.dist)):
                            temp_max_index =temp_neighbor.dist.index(max(temp_neighbor.dist)) ;
                            temp_neighbor.dist[temp_max_index] = dist ;
                            temp_neighbor.index[temp_max_index] = j ;
            self.nearest_neighbor_list.append(temp_neighbor) ;
    def prediction_angle(self):
        self.variance_cos = [] ;
        for i in range(self.data_length):
            temp_neighbor_index_list = [] ;
            temp_neighbor_index_list = self.nearest_neighbor_list[i].index ;
            vector_space = [] ; # store vector's in temp_neighbor_index_list
            for j in range(len(temp_neighbor_index_list)):
                point_1_index = temp_neighbor_index_list[j] ;
                point_1 = self.format_x[point_1_index] ;
                for k in range( j + 1, len(temp_neighbor_index_list), 1):
                    try:
                        point_2_index = temp_neighbor_index_list[k] ;
                        point_2 = self.format_x[point_2_index] ;
                    except:
                        print("point_2_index", point_2_index) ;
                        print("data_length", self.data_length) ;
                    vector = self.vector_minus(point_1, point_2) ;
                    vector_space.append(vector) ;
            # use vector space to count cos
            temp_vector_cos = self.count_cos(vector_space, i) ;
            var = statistics.variance(temp_vector_cos) ;
            self.variance_cos.append(var) ;
            
    def count_cos(self, vector_space, round):
        temp_vector_cos = [] ;
        for i in range(len(vector_space)):
            vector_1 = vector_space[i] ;
            for j in range( i+1, len(vector_space), 1):
                vector_2 = vector_space[j] ;

                inner_product = self.vector_inner_product(vector_1, vector_2) ;

                vector_1_pow = self.vector_pow(vector_1) ;
                if(vector_1_pow == 0):
                    vector_1_pow = 0.000000000000001 ;
                    self.error_1 = self.error_1 + 1 ;
                    #print("vector_1 neighbor_index", self.nearest_neighbor_list[i].index) ;
                vector_2_pow = self.vector_pow(vector_2) ;
                if(vector_2_pow == 0):
                    vector_2_pow = 0.000000000000001 ; 
                    self.error_2 = self.error_2 + 1 ;
                    #print("vector_2 neighbor_index", self.nearest_neighbor_list[i].index) ;
                if(vector_1_pow == 0 and vector_2_pow == 0):
                    self.error_12 = self.error_12 + 1 ;
                    #print("vector_12 neighbor_index", self.nearest_neighbor_list[i].index) ;
                #print("vector_1", vector_1) ;
                #print("vector_2", vector_2) ;
                cos = inner_product/(vector_1_pow*vector_2_pow) ;
                temp_vector_cos.append(cos) ;
        return temp_vector_cos ;

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

    class neighbor:
        dist = [] ;
        index = [] ;
        def refresh(self):
            self.dist = [] ;
            self.index = [] ;

class Gen:
    record_model = [] ;
    def get_record_model(self, data_name):
        self.record_model = [] ;
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
                    self.record_model.append(temp_record_model) ;
class Validation:
    train_x_copy= [] ;
    train_y_copy = [] ;
    train_validation_x = [] ;
    train_validation_y = [] ;
    train_less_x = [] ;
    train_less_y = [] ;
    part = 0 ;
    def __init__(self, train_x_copy, train_y_copy , part) :
        self.train_x_copy = copy.deepcopy(train_x_copy) ;
        self.train_y_copy = copy.deepcopy(train_y_copy) ;
        self.part = part ;
    def main_control(self):
        self.cut_down() ;

    def cut_down(self):
        denominator_list = [] ;
        
        for i in range(self.part):
            denominator_list.append(i) ;
        for i in range(self.part):
            temp_train_validation_x = [] ;
            temp_train_validation_y = [] ;
            temp_train_less_x = [] ;
            temp_train_less_y = [] ;
            for j in range(len(self.train_x_copy)):
                if( (j%self.part) == denominator_list[i]):
                    temp_train_validation_x.append(self.train_x_copy[j]) ;
                    temp_train_validation_y.append(self.train_y_copy[j]) ;
                else:
                    temp_train_less_x.append(self.train_x_copy[j]) ;
                    temp_train_less_y.append(self.train_y_copy[j]) ;
            self.train_validation_x.append(temp_train_validation_x) ;
            self.train_validation_y.append(temp_train_validation_y) ;
            self.train_less_x.append(temp_train_less_x) ;
            self.train_less_y.append(temp_train_less_y) ;


        """
        data_validation_length = len(self.train_x_copy)/self.part ;
        flag_train_x_copy = 0 ;
        for i in range(self.part):
            temp_train_validation_x = [] ;
            temp_train_validation_y = [] ;
            temp_train_less_x = [] ;
            temp_train_less_y = [] ;
            temp_mark_validation = [] ;
            flag_train_validation = 0 ;
            flag_less = 0 ;
            for j in range(flag_train_x_copy, flag_train_x_copy+data_validation_length, 1):
                if( j >= len(self.train_x_copy)):
                    break ;
                temp_train_validation_x.append(self.train_x_copy[j]) ;
                temp_train_validation_y.append(self.train_y_copy[j]) ;
                temp_mark_validation.append(j) ;
            for j in range(len(self.train_x_copy)):
                if( j in temp_mark_validation):
                    continue;
                temp_train_less_x.append(self.train_x_copy[j]) ;
                temp_train_less_y.append(self.train_y_copy[j]) ;
            flag_train_x_copy = flag_train_x_copy + data_validation_length ;
            self.train_validation_x.append(temp_train_validation_x) ;
            self.train_validation_y.append(temp_train_validation_y) ;
            self.train_less_x.append(temp_train_less_x) ;
            self.train_less_y.append(temp_train_less_y) ;
            """
class ROF_UnLabeled_Method:
    train_x_copy= [] ;
    train_y_copy = [];
    CR_List = [] ; # record every round of clusterSize and ROF value
    nr_each_label = [] ;
    resolution_max = 0.0 ;
    resolution_min = 0.0 ;
    current_r = 0.0 ;
    delta_r_percent = 0.04;# 4% - 10% 
    nr_topN = 0 ;
    All_Is_Cluster = False ;
    number_of_point = 0 ;
    number_of_unmerged_point = 1000 ;
    nr_unmerge_list = [] ;

    def __init__(self, train_x_copy, train_y_copy, top_n) :
        self.train_x_copy = [] ;
        self.train_y_copy = [] ;
        self.train_x_copy = copy.deepcopy(train_x_copy) ;
        self.train_y_copy = copy.deepcopy(train_y_copy) ;
        self.nr_topN = top_n ;
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
        self.delta_r_percent = 0.1 ;# 4% - 10%  a1a:0.25
        self.All_Is_Cluster = False ;
        self.number_of_unmerged_point = 1000 ;
        self.nr_unmerge_list = [] ;
        
        for i in range(self.number_of_point):
            self.nr_each_label.append(1) ;

    def calculate_resolution_range(self):
        #max_min_neiDist = 0.0 ;
        #min_min_neiDist = 0.0 ;
        min_neiDist = [] ;
        for i in range(len(self.train_x_copy)):# 1605
            minimum = 1000000000.0 ;
            for j in range(len(self.train_x_copy)):
                if( i != j ):
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
        #print("resolution_max", self.resolution_max) ;
        #print("resolution_min", self.resolution_min) ;
        self.current_r = self.resolution_max ;
        #print("before while current_r", self.current_r) ;
        self.All_Is_Cluster = True ;
        round = 1 ;
        while  (self.number_of_unmerged_point > self.nr_topN): # (self.All_Is_Cluster == True) and (self.number_of_unmerged_point > self.nr_topN)
        #while(current_r>=self.resolution_min):
            self.All_Is_Cluster = False ;
            self.current_r = self.current_r - (self.current_r - self.resolution_min) * self.delta_r_percent ; # current_r = current_r - (current_r - self.resolution_min) * self.delta_r_percent   # self.resolution_max
            #print("\n") ;
            #print("round",round) ;
            #print("current_r", self.current_r) ;
            self.RB_CLUSTER( round, self.current_r) ;
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
            #Merged_point = False ;
            for j in range(i, self.number_of_point, 1):
                if( self.CR_List[round].label[i] != self.CR_List[round].label[j]  and i != j ):
                    temp_dist = self.euc_dist(self.train_x_copy[i], self.train_x_copy[j]) ;# euclidean distance ; maybe should try maha distance
                    if(temp_dist<=1):
                        self.All_Is_Cluster = True ;
                        #Merged_point = True ;
                        if(self.CR_List[round].label[i]<self.CR_List[round].label[j]):# update label
                            self.CR_List[round].label[j] = self.CR_List[round].label[i] ;
                            self.nr_each_label[i] = self.nr_each_label[i] + 1 ;
                        else:
                            self.CR_List[round].label[i] = self.CR_List[round].label[j] ;
                            self.nr_each_label[j] = self.nr_each_label[j] + 1 ;
                        #self.CR_List[round].ClusterSize[i] = self.CR_List[round].ClusterSize[i] + 1 ;
                        #self.CR_List[round].ClusterSize[j] = self.CR_List[round].ClusterSize[j] + 1 ;
            """
            if(Merged_point == False): # record number_of_unmerged_point 
                self.number_of_unmerged_point = self.number_of_unmerged_point + 1 ;
            """

        # synchronized each point's ClusterSize
        for i in range(self.number_of_point):
            label = self.CR_List[round].label[i] ;
            self.CR_List[round].ClusterSize[i] = self.nr_each_label[label] ;
        # count unmerged points
        for i in range(self.number_of_point):
            temp_size = self.CR_List[round].ClusterSize[i] ;
            if(temp_size == 1):
                self.number_of_unmerged_point = self.number_of_unmerged_point + 1 ;
        print("number_of_unmerged_point", self.number_of_unmerged_point) ;
        self.nr_unmerge_list.append(self.number_of_unmerged_point) ; # record nr_unmerge_point
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
class KNN:
    train_x_copy= [] ;
    train_y_copy = [] ;
    neighbor_list = [] ; # record nearest neighbor index
    wrong_percent_list = [] ;
    data_length = 0 ;
    k = 0 ;
    threshold = 0.0 ;
    def __init__(self, train_x_copy, train_y_copy , k, p): # find nearest k neighbor and decide p threshold
        self.train_x_copy = [] ;
        self.train_y_copy = [] ;
        self.train_x_copy = copy.deepcopy(train_x_copy) ;
        self.data_length = len(self.train_x_copy) ;
        self.train_y_copy = copy.deepcopy(train_y_copy) ;
        self.k = k ;
        self.threshold = p ;
    def main_control(self):
        self.refresh() ;
        self.find_nearest_neighbor() ;
        self.compare_neighbor() ;
        self.record_percentage() ;
    def refresh(self):
        self.neighbor_list = [] ;
        self.wrong_percent_list = [] ;
    def find_nearest_neighbor(self):
        for i in range(self.data_length):
            temp_neighbor_dist = [] ;
            temp_neighbor_list = [] ;

            for j in range(self.data_length):
                if( i != j ):
                    temp = [self.train_x_copy[i] , self.train_x_copy[j]] ;
                    words = sorted(list(reduce(set.union, map(set, temp)))) ;
                    feats = zip(*[[d.get(w, 0) for d in temp] for w in words]) ;
                    #dst = distance.euclidean(feats[0],feats[1]) ;
                    dst = distance.cityblock(feats[0],feats[1]) ;
                    
                   # dst = self.euc_dist(self.train_x_copy[i], self.train_x_copy[j]) ;
                    if(len(temp_neighbor_list)<self.k):
                        temp_neighbor_dist.append(dst) ;
                        temp_neighbor_list.append(j) ;
                    else:
                        if(dst<max(temp_neighbor_dist)):
                            temp_max_index = temp_neighbor_dist.index(max(temp_neighbor_dist)) ;
                            temp_neighbor_dist[temp_max_index] = dst ;
                            temp_neighbor_list[temp_max_index] = j ;
            self.neighbor_list.append(temp_neighbor_list) ;
    def compare_neighbor(self):
        for i in range(self.data_length):
            q_list = [] ;
            n_list = [] ;
            temp_neighbor_list = [] ;
            temp_neighbor_list = self.neighbor_list[i] ;
            print("i", i) ;
            print("temp_neighbor_list", temp_neighbor_list) ;
            label_q = self.train_y_copy [i] ;
            count = 0.0 ;
            for j in range(self.k):
                neighbor_index = temp_neighbor_list[j] ;
                label_n = self.train_y_copy[j] ;
                if(label_q != label_n):
                    q_list.append(label_q) ;
                    n_list.append(label_n) ;
                    count = count + 1 ;
            percentage_of_difference = 0.0 ;
            percentage_of_difference = count/(float)(self.k) ;
            print("q_list", q_list) ;
            print("n_list", n_list) ;
            print("count", count) ;
            print("percentage_of_difference", percentage_of_difference) ;
            #my idea !!!!!!!!  I use percentage_of_difference instead of threshold, therefore I could pick the maximum value of wrong_percent_list as my outlier
            self.wrong_percent_list.append(percentage_of_difference) ;
            #!!!!
    def record_percentage(self):
        upper_one = 0 ;
        upper_dot_nine = 0 ;
        upper_dot_six= 0 ;
        upper_dot_four = 0 ;
        upper_dot_two = 0 ;
        upper_dot_zero = 0 ;
        for i in range(len(self.wrong_percent_list)):
            temp_percent = self.wrong_percent_list[i]
            if(temp_percent >= 1.0):
                upper_one = upper_one + 1 ;
            elif(temp_percent >= 0.9):
                upper_dot_nine = upper_dot_nine + 1 ;
            elif(temp_percent >= 0.6):
                upper_dot_six = upper_dot_six + 1 ;
            elif(temp_percent >= 0.4):
                upper_dot_four = upper_dot_four + 1 ;
            elif(temp_percent >= 0.2):
                upper_dot_two = upper_dot_two + 1 ;
            else:
                upper_dot_zero = upper_dot_zero + 1 ;
        print("upper_one", upper_one) ;
        print("upper_dot_nine", upper_dot_nine) ;
        print("upper_dot_six", upper_dot_six) ;
        print("upper_dot_four", upper_dot_four) ;
        print("upper_dot_two", upper_dot_two) ;
        print("upper_dot_zero", upper_dot_zero) ;
    def euc_dist(self, a, b):
        temp = 0.0 ;
        for i in range(len(a)):
            temp = temp + ((a[i] - b[i]) * (a[i] - b[i])) ;
        temp = math.pow(temp, 0.5) ;
        return temp ;
class Horizontal_cross_validation:
    data_name = "" ;
    train_x_copy= [] ;
    train_y_copy = [] ;
    weight_list = [] ;
    weight_acc_list = [] ;
    final_acc_list = [] ;
    final_weight = [] ;
    block = 0 ;
    base_size = 0 ;
    
    all_valid_x = [] ;
    all_valid_y = [] ;
    all_less_x = [] ;
    all_less_y = [] ;
    def __init__(self, data_name, train_x_copy, train_y_copy , weight_list, block, base_size_percentage): # 
        self.data_name = data_name ;
        self.train_x_copy = copy.deepcopy(train_x_copy) ;
        self.train_y_copy = copy.deepcopy(train_y_copy) ;
        self.weight_list = copy.deepcopy(weight_list) ;
        self.block = block ;
        self.base_size = (int)(len(self.train_x_copy) * base_size_percentage) ;
        valid = Validation(self.train_x_copy, self.train_y_copy, block) ;
        valid.main_control() ;
        self.all_valid_x = copy.deepcopy(valid.train_validation_x) ;
        self.all_valid_y = copy.deepcopy(valid.train_validation_y) ;
        self.all_less_x = copy.deepcopy(valid.train_less_x) ;
        self.all_less_y = copy.deepcopy(valid.train_less_y) ;

    def main_control(self):
        self.refresh() ;
        self.cross_weight() ;
        self.make_final_weight() ;
    def refresh(self):
        self.weight_acc_list = [] ;
        self.final_acc_list = [] ;
        self.final_weight = [] ;
    def cross_weight(self):
        for i in range(self.block):
            mo_list = [] ;
            temp_weight_acc_list = [] ;
            valid_x = copy.deepcopy(self.all_valid_x[i]) ;
            valid_y = copy.deepcopy(self.all_valid_y[i]) ;
            less_x = copy.deepcopy(self.all_less_x[i]) ;
            less_y = copy.deepcopy(self.all_less_y[i]) ;
            less_data_length = len(less_x) ;
                
            mo = Method_order() ;
            mo_list.append(mo.Angle_Based(less_x, less_y, self.data_name));
            mo_list.append(mo.SVM_Out(less_x, less_y, self.data_name));
            mo_list.append(mo.Rof_Out(less_x, less_y, self.data_name, self.base_size)) ;
            mo_list.append(mo.Lof_Out(less_x, less_y, self.data_name)) ;
            for j in range(len(self.weight_list)):
                temp_hybrid_order = [] ;
                temp_weight = self.weight_list[j] ;
                temp_hybrid_order = self.make_hybrid_order(temp_weight, mo_list, less_data_length) ;
                accuracy = self.calculate_accuracy(valid_x, valid_y, less_x, less_y, temp_hybrid_order) ;
                temp_weight_acc_list.append(accuracy) ;
            self.weight_acc_list.append(temp_weight_acc_list) ;
       
    def make_hybrid_order(self, temp_weight, method_order, data_length):
        temp_hybrid_order = [] ;
        for j in range(data_length):
            temp_hybrid_order.append(0.0) ;
        for j in range(data_length):
            for k in range(len(method_order)):
                temp_hybrid_order[j] = temp_hybrid_order[j] + temp_weight[k] * method_order[k].index(j) ;
        return temp_hybrid_order ;
    def calculate_accuracy(self, valid_x, valid_y, less_x, less_y, temp_hybrid_order):
        valid_x = copy.deepcopy(valid_x) ;
        valid_y = copy.deepcopy(valid_y) ;
        less_x = copy.deepcopy(less_x) ;
        less_y = copy.deepcopy(less_y) ;
        temp_hybrid_order = copy.deepcopy(temp_hybrid_order) ;

        for i in range(self.base_size):
            pop_index = temp_hybrid_order.index(min(temp_hybrid_order)) ;
            temp_hybrid_order.pop(pop_index) ;
            less_x.pop(pop_index) ;
            less_y.pop(pop_index) ;
        write_svm_file (less_x , less_y , self.data_name+'-temp3.train') ;
        cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + self.data_name+'-temp3.train.model ' + self.data_name+'-temp3.train' ;
        subprocess.call (cmd.split()) ;
        m2 = load_model (self.data_name+'-temp3.train.model') ;
        p_label , p_acc , p_val = predict(valid_y , valid_x , m2) ;
        return p_acc[0] ;
    def make_final_weight(self):
        for i in range(len(self.weight_acc_list[0])):
            sum = 0.0 ;
            avg = 0.0 ;
            for j in range(len(self.weight_acc_list)):
                sum = sum + self.weight_acc_list[j][i] ;
            avg = sum/(float)(self.block) ;
            self.final_acc_list.append(avg) ;
        index = self.final_acc_list.index(max(self.final_acc_list)) ;
        self.final_weight = self.weight_list[index] ;
        
    def make_final_hybrid_order(self, train_x_copy, train_y_copy, final_weight, base_size):
        hybrid_order = [] ;
        mo_list = [] ;
        mo = Method_order() ;
        mo_list.append(mo.Angle_Based(train_x_copy, train_y_copy, self.data_name));
        mo_list.append(mo.SVM_Out(train_x_copy, train_y_copy, self.data_name));
        mo_list.append(mo.Rof_Out(train_x_copy, train_y_copy, self.data_name, base_size)) ;
        mo_list.append(mo.Lof_Out(train_x_copy, train_y_copy, self.data_name)) ;
        hybrid_order = self.make_hybrid_order(final_weight, mo_list, len(train_x_copy)) ;#(self, temp_weight, method_order, data_length)
        return hybrid_order ;


        
class Method_order:
    def Angle_Based(self, train_x_copy_1, train_y_copy_1 ,data_name):
        train_x_copy = copy.deepcopy(train_x_copy_1) ;
        train_y_copy = copy.deepcopy(train_y_copy_1) ;
        index_order = [] ;
        instance_order = [] ;
        soa = Gen() ;
        soa.get_record_model(data_name) ;
        gen =  General(train_x_copy, soa.record_model) ;
        gen.main_control() ;
        parameter_k =5 ;
        svm_outlier_angle = Angle_Based_FastABOD(train_x_copy, gen.format_x, train_y_copy, parameter_k) ;
        svm_outlier_angle.main_control() ;
        for i in range(len(svm_outlier_angle.variance_cos)):
            svm_outlier_angle.variance_cos[i] = svm_outlier_angle.variance_cos[i] + 0.0000000001*i ;
        variance_cos = copy.deepcopy(svm_outlier_angle.variance_cos) ;
        data_length = len(svm_outlier_angle.variance_cos) ;

        for i in range(data_length):
            temp_index = svm_outlier_angle.variance_cos.index(min(svm_outlier_angle.variance_cos)) ;
            temp_instance = svm_outlier_angle.variance_cos.pop(temp_index) ;
            instance_order.append(temp_instance) ;

        for i in range(len(instance_order)):
            temp_instance = instance_order[i] ;
            index = variance_cos.index(temp_instance) ;
            index_order.append(index) ;
        return index_order ;
    def SVM_Out(self, train_x_copy_1, train_y_copy_1 , data_name):
        train_x_copy = copy.deepcopy(train_x_copy_1) ;
        train_y_copy = copy.deepcopy(train_y_copy_1) ;
        index_order = [];
        instance_order = [] ;
        svo = Gen() ;
        svo.get_record_model(data_name) ;
        svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, svo.record_model) ;
        svm_outlier.main_control() ;

        for i in range(len(svm_outlier.compare_distance)):
            svm_outlier.compare_distance[i] = svm_outlier.compare_distance[i] +  0.0000000001*i ;
        compare_distance = copy.deepcopy(svm_outlier.compare_distance) ;
        data_length = len(svm_outlier.compare_distance) ;

        for i in range(data_length):
            temp_index = svm_outlier.compare_distance.index(min(svm_outlier.compare_distance)) ;# min
            temp_instance = svm_outlier.compare_distance.pop(temp_index) ;
            instance_order.append(temp_instance) ;

        for i in range(len(instance_order)):
            temp_instance = instance_order[i] ;
            index = compare_distance.index(temp_instance) ;
            index_order.append(index) ;
        return index_order ;
    def Rof_Out(self, train_x_copy_1, train_y_copy_1, data_name, base_size):
        CR = [] ;
        train_x_copy = copy.deepcopy(train_x_copy_1) ;
        train_y_copy = copy.deepcopy(train_y_copy_1) ;
        index_order = [];
        instance_order = [] ;

        svr = Gen() ;
        svr.get_record_model(data_name) ;

        svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, svr.record_model) ;
        svm_outlier.main_control() ;

        gen =  General(train_x_copy, svr.record_model) ;
        gen.main_control() ;

        rof = ROF_Method(gen.format_x, train_y_copy, base_size) ;
        rof.main_control() ;
        for i in range(len(rof.CR_List[len(rof.CR_List)-1].ROF)):
            rof.CR_List[len(rof.CR_List)-1].ROF[i] = rof.CR_List[len(rof.CR_List)-1].ROF[i] + 0.000000001*i; # for order convenience random.uniform(0.1,0.001)

        CR = copy.deepcopy(rof.CR_List[len(rof.CR_List)-1].ROF) ;
        data_length = len(rof.CR_List[len(rof.CR_List)-1].ROF) ;

        for i in range(data_length):
            temp_instance = min(rof.CR_List[len(rof.CR_List)-1].ROF) ;
            temp_index = rof.CR_List[len(rof.CR_List)-1].ROF.index(temp_instance) ;
            rof.CR_List[len(rof.CR_List)-1].ClusterSize.pop(temp_index) ;
            rof.CR_List[len(rof.CR_List)-1].label.pop(temp_index) ;
            rof.CR_List[len(rof.CR_List)-1].ROF.pop(temp_index) ;
            instance_order.append(temp_instance) ;

        for i in range(len(instance_order)):
            temp_instance = instance_order[i] ;
            index = CR.index(temp_instance) ;
            index_order.append(index) ;
            #!!!!!!!!!!!!!!!!!!!! WRONG 
        return index_order ;


    def Lof_Out(self, train_x_copy_1, train_y_copy_1, data_name):
        train_x_copy = copy.deepcopy(train_x_copy_1) ;
        train_y_copy = copy.deepcopy(train_y_copy_1) ;
        index_order = [];
        instance_order = [] ;
        lof = Gen() ;
        lof.get_record_model(data_name) ;

        LOF_point = LOF(train_x_copy, train_y_copy, lof.record_model, K, MinPts) ; 
        LOF_point.main_control() ; 
        for i in range(len(LOF_point.all_LOF_list)):
            LOF_point.all_LOF_list[i] = LOF_point.all_LOF_list[i] + 0.00000001*i; # for order convenience
        lof_list = copy.deepcopy(LOF_point.all_LOF_list) ;

        for i in range(len(LOF_point.all_LOF_list)):
            temp_index = LOF_point.all_LOF_list.index(max(LOF_point.all_LOF_list)) ;#max
            temp_instance = LOF_point.all_LOF_list.pop(temp_index) ;
            instance_order.append(temp_instance) ;
        for i in range(len(instance_order)):
            temp_instance = instance_order[i] ;
            index = lof_list.index(temp_instance) ;
            index_order.append(index) ;
        return index_order ;


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

def run (data_name , pickle_path ,select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) :
    train_x_copy = copy.deepcopy(train_x) ;
    train_y_copy = copy.deepcopy(train_y) ;

    train_x_copy_o = copy.deepcopy(train_x) ;
    train_y_copy_o = copy.deepcopy(train_y) ;
    
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
        # hybrid

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

        pickle_path = pickle_path + '_svm_angle' ;
      #SOA_VC = copy.deepcopy(svm_outlier_angle.variance_cos) ;
        """
        record_model_A = [] ;
        svm_outlier_angle_order = [] ;
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
                record_model_A.append(temp_record_model) ;
            svm_outlier_angle_pickle = SVM_Outlier_Angle(train_x_copy_o, train_y_copy_o, record_model_A) ;
            svm_outlier_angle_pickle.main_control() ;

        for i in range(len(train_y_copy)):
            if(len(svm_outlier_angle_pickle.variance_cos)>0):
                temp = svm_outlier_angle_pickle.variance_cos.index(min(svm_outlier_angle_pickle.variance_cos)) ;
                temp_pop_index = svm_outlier_angle_pickle.wrong_label_index[temp] ;
                    
                svm_outlier_angle_pickle.variance_cos.pop(temp) ;

                train_x_copy_o.pop(temp_pop_index) ;
                train_y_copy_o.pop(temp_pop_index) ;
                svm_outlier_angle_order.append(temp_pop_index) ;
          """



        for i in range (data_num_size) :
            soa = Gen() ;
            soa.get_record_model(data_name) ;
            #print("i", i) ;
            #print("train_x_copy length:", len(train_x_copy)) ;
            gen =  General(train_x_copy, soa.record_model) ;
            gen.main_control() ;
            #print("gen.format_x length:", len(gen.format_x)) ;

            parameter_k =5 ;
            svm_outlier_angle = Angle_Based_FastABOD(train_x_copy, gen.format_x, train_y_copy, parameter_k) ;
            svm_outlier_angle.main_control() ;
            #print("error_1", svm_outlier_angle.error_1) ;
            #print("error_2", svm_outlier_angle.error_2) ;
            #print("error_12", svm_outlier_angle.error_12) ;

            for j in range (base_size) :
    
                #svm_outlier_angle = SVM_Outlier_Angle(train_x_copy, train_y_copy, record_model) ;
                #svm_outlier_angle.main_control() ;
                if(len(svm_outlier_angle.variance_cos)>0):
                    temp = svm_outlier_angle.variance_cos.index(min(svm_outlier_angle.variance_cos)) ;
                    #temp_pop_index = svm_outlier_angle.wrong_label_index[temp] ;
                    #svm_outlier_angle.format_x.pop(temp) ;
                    svm_outlier_angle.variance_cos.pop(temp) ;

                    train_x_copy.pop(temp) ;
                    train_y_copy.pop(temp) ;

            write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
            cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
            subprocess.call (cmd.split()) ;
            m2 = load_model (data_name+'-temp2.train.model') ;
            # m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
            
            p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
            E_in.append(p_acc[0]/base_acc_in) ;
            p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
            E_out.append(p_acc[0]/base_acc_out) ;
            svm_outlier_angle.refresh() ;    

        """
        with open(pickle_path, 'w') as store:
             pickle.dump((pure_data_name, svm_outlier_angle_order, E_out, E_in), store) ;
             print("dd") ;
        """
        
    
    elif (select_method == 3) :
        pickle_path = pickle_path + '_svm_dist' ;
      #SOA_VC = copy.deepcopy(svm_outlier_angle.variance_cos) ;
        """
        record_model_A = [] ;
        svm_outlier_dist_order = [] ;
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
                record_model_A.append(temp_record_model) ;
            svm_outlier_angle_pickle = SVM_Outlier(train_x_copy_o, train_y_copy_o, record_model_A) ;
            svm_outlier_angle_pickle.main_control() ;
        for i in range(len(train_y_copy)):
            temp_pop_index = svm_outlier_angle_pickle.compare_distance.index(min(svm_outlier_angle_pickle.compare_distance)) ;# min

            svm_outlier_angle_pickle.compare_distance.pop(temp_pop_index) ;

            train_x_copy_o.pop(temp_pop_index) ;
            train_y_copy_o.pop(temp_pop_index) ;
            svm_outlier_dist_order.append(temp_pop_index) ;

        """
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
        with open(pickle_path, 'w') as store:
             pickle.dump((pure_data_name, svm_outlier_dist_order, E_out, E_in), store) ;
        """
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
        """
        pickle_path = pickle_path + 'rof' ;
        record_model_A = [] ;
        rof_order = [] ;
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
                record_model_A.append(temp_record_model) ;
        svm_outlier = SVM_Outlier(train_x_copy_o, train_y_copy_o, record_model_A) ;
        svm_outlier.main_control() ;

        gen_A =  General(train_x_copy_o, record_model_A) ;
        gen_A.main_control() ;
        print("OUT_train_x,_copy" , len(gen_A.format_x)) ;
        print("OUT_train_y_copy" , len(train_y_copy_o)) ;

        rof = ROF_Method(gen_A.format_x, train_y_copy_o) ;
        rof.main_control() ;


        for i in range(data_num_size * base_size):
            ROF_min = min(rof.CR_List[len(rof.CR_List)-1].ROF) ;
            temp_int = rof.CR_List[len(rof.CR_List)-1].ROF.index(ROF_min) ;
            rof.CR_List[len(rof.CR_List)-1].ClusterSize.pop(temp_int) ;
            rof.CR_List[len(rof.CR_List)-1].label.pop(temp_int) ;
            rof.CR_List[len(rof.CR_List)-1].ROF.pop(temp_int) ;
            print("ROF_min", ROF_min) ;
                
            print("resolution_min", rof.resolution_min) ;
            print("resolution_max", rof.resolution_max) ;
            train_x_copy_o.pop(temp_int) ;
            train_y_copy_o.pop(temp_int) ;
            rof_order.append(temp_int) ;
        """
        for i in range (data_num_size) :#data_num_size
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

            rof = ROF_Method(gen.format_x, train_y_copy, base_size) ;
            rof.main_control() ;
            #testing!!!!!!!!!!!!!
            temp_count = 0 ;
            for k in range(len(rof.CR_List[len(rof.CR_List)-1].ROF)):
                temp = rof.CR_List[len(rof.CR_List)-1].ROF[k] ;
                if(temp != 0):
                    #print(temp) ;
                    temp_count = temp_count + 1 ;
            print("original_length", len(rof.CR_List[len(rof.CR_List)-1].ROF)) ;
            print("no_zero_length", temp_count) ;
            for j in range (base_size) :#base_size
                ROF_min = min(rof.CR_List[len(rof.CR_List)-1].ROF) ;
                temp_int = rof.CR_List[len(rof.CR_List)-1].ROF.index(ROF_min) ;
                rof.CR_List[len(rof.CR_List)-1].ClusterSize.pop(temp_int) ;
                rof.CR_List[len(rof.CR_List)-1].label.pop(temp_int) ;
                rof.CR_List[len(rof.CR_List)-1].ROF.pop(temp_int) ;
                #print("ROF_min", ROF_min) ;
                
                #print("resolution_min", rof.resolution_min) ;
                #print("resolution_max", rof.resolution_max) ;
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
        with open(pickle_path, 'w') as store:
             pickle.dump((pure_data_name, rof_order, E_out, E_in), store) ;
        """
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
        for i in range (data_num_size) :
            lof = Gen() ;
            lof.get_record_model(data_name) ;

            LOF_point = LOF(train_x_copy, train_y_copy, lof.record_model, K, MinPts) ; 
            LOF_point.main_control() ; 
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

        block = 5 ;
        
        for k in range (data_num_size) :
            valid = Validation(train_x_copy, train_y_copy, block) ;
            valid.main_control() ;
            weight_list = [] ;
            final_weight = [] ;
            method_order = [] ;
            
            for i in range(block):
                valid_x = valid.train_validation_x[i] ;
                valid_y =valid.train_validation_y[i] ;
                less_x = valid.train_less_x[i] ;
                less_y = valid.train_less_y[i] ;
                base_valid_size = len(less_x) * 0.01 ;# base_valid_size decide the pop size accuracy in validation
                base_valid_size = int(base_valid_size) + 1 ;
                temp_weight = cross_validation( data_name , "T T" , data_num_size , base_valid_size , base_acc_in , base_acc_out , less_x , less_y , valid_x , valid_y ) ;
                weight_list.append(temp_weight) ;
            print("weight_list", weight_list) ;
            for i in range(len(weight_list[0])):
                temp_total = 0.0 ;
                for j in range(len(weight_list)):
                    temp_total = temp_total + weight_list[j][i] ;
                temp_total = temp_total/block ;
                final_weight.append(temp_total) ;
            
            #final_weight = [0.18, 0.36, 0.24, 0.22] ;
            print("final_weight", final_weight) ;
            angle_base_order = Angle_Based(train_x_copy, train_y_copy, data_name) ;
            svm_outlier_order = SVM_Out(train_x_copy, train_y_copy, data_name) ;
            rof_order = Rof_Out(train_x_copy, train_y_copy, data_name, base_size) ;
            lof_order = Lof_Out(train_x_copy, train_y_copy, data_name) ;
            method_order.append(angle_base_order) ;
            method_order.append(svm_outlier_order) ;
            method_order.append(rof_order) ;
            method_order.append(lof_order) ;
            data_length = len(train_x_copy) ;
            All_weight = [] ;
            All_weight.append(final_weight) ;
            temp_hybrid_order = [] ;
            hybrid_order = [] ;
            temp_hybrid_order = Make_hybrid_order(base_size, All_weight, method_order, data_length) ;
            print("length temp_hybrid_order", len(temp_hybrid_order)) ;
            hybrid_order = copy.deepcopy(temp_hybrid_order[0]) ;
            for j in range (base_size) :
                temp_pop_index = hybrid_order.index(min(hybrid_order)) ;#max
                #temp_pop_index = dst_list.index(max(dst_list)) ;
                hybrid_order.pop(temp_pop_index) ;
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
        block = 5 ;
        valid = Validation(train_x_copy, train_y_copy, block) ;
        valid.main_control() ;
        weight_list = [] ;
        final_weight = [] ;
        for i in range(block):
            valid_x = valid.train_validation_x[i] ;
            valid_y =valid.train_validation_y[i] ;
            less_x = valid.train_less_x[i] ;
            less_y = valid.train_less_y[i] ;
            base_valid_size = len(less_x) * 0.05 ;# base_valid_size decide the pop size accuracy in validation
            base_valid_size = int(base_valid_size) + 1 ;
            temp_weight = cross_validation( data_name , "T T" , data_num_size , base_valid_size , base_acc_in , base_acc_out , less_x , less_y , valid_x , valid_y ) ;
            weight_list.append(temp_weight) ;
        print("weight_list", weight_list) ;
        for i in range(len(weight_list[0])):
            temp_total = 0.0 ;
            for j in range(len(weight_list)):
                temp_total = temp_total + weight_list[j][i] ;
            temp_total = temp_total/block ;
            final_weight.append(temp_total) ;
        print("final_weight", final_weight) ;
        """
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
        percentage = 0.12 ; # decide unsupervised method outlier
        restriction = 0 ; # decided from ROF nr_unmerge_list
        """
        svr = Gen() ;
        svr.get_record_model(data_name) ;
        gen =  General(train_x_copy, svr.record_model) ;
        gen.main_control() ;
        rof = ROF_UnLabeled_Method(gen.format_x, train_y_copy, base_size) ;
        rof.main_control() ;
        
        const_restriction = (int)((len(train_x_copy)) * percentage) ;
        for i in range(len(rof.nr_unmerge_list)):
            temp_nr_unmerge = rof.nr_unmerge_list[i] ;
            if(temp_nr_unmerge<const_restriction):
                restriction = temp_nr_unmerge ;
                break ;
        """
        for i in range (data_num_size) :
            
            if(i<=-1): # i<=10 restriction>0
                print("data_num_size", data_num_size) ;
                svr = Gen() ;
                svr.get_record_model(data_name) ;
                gen =  General(train_x_copy, svr.record_model) ;
                gen.main_control() ;
                rof = ROF_UnLabeled_Method(gen.format_x, train_y_copy, base_size) ;
                rof.main_control() ;
                #testing!!!!!!!!!!!!!
                temp_count = 0 ;
                for k in range(len(rof.CR_List[len(rof.CR_List)-1].ROF)):
                    temp = rof.CR_List[len(rof.CR_List)-1].ROF[k] ;
                    if(temp != 0):
                        #print(temp) ;
                        temp_count = temp_count + 1 ;
                print("original_length", len(rof.CR_List[len(rof.CR_List)-1].ROF)) ;
                print("no_zero_length", temp_count) ;
            else:
                knn = KNN( train_x_copy, train_y_copy, 10, 0.8) ; # train_x_copy
                knn.main_control() ;
            for j in range (base_size) :
                if(i<=-1): # restriction>0
                    ROF_min = min(rof.CR_List[len(rof.CR_List)-1].ROF) ;
                    temp_int = rof.CR_List[len(rof.CR_List)-1].ROF.index(ROF_min) ;
                    rof.CR_List[len(rof.CR_List)-1].ClusterSize.pop(temp_int) ;
                    rof.CR_List[len(rof.CR_List)-1].label.pop(temp_int) ;
                    rof.CR_List[len(rof.CR_List)-1].ROF.pop(temp_int) ;
                    restriction = restriction - 1 ;
                else:
                    KNN_max = max(knn.wrong_percent_list) ;
                    print(KNN_max) ;
                    temp_int = knn.wrong_percent_list.index(KNN_max) ;
                    knn.wrong_percent_list.pop(temp_int) ;

                try:
                    train_x_copy.pop(temp_int) ;
                    train_y_copy.pop(temp_int) ;
                except:
                    print("length wrong_percent_list", len(knn.wrong_percent_list)) ;
                    print("temp_int", temp_int) ;
                    
                

            write_svm_file (train_x_copy , train_y_copy , data_name+'-temp2.train') ;
            cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp2.train.model ' + data_name+'-temp2.train' ;
            subprocess.call (cmd.split()) ;
            m2 = load_model (data_name+'-temp2.train.model') ;
            # m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
            
            p_label , p_acc , p_val = predict(train_y , train_x , m2) ;
            E_in.append(p_acc[0]/base_acc_in) ;
            p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
            E_out.append(p_acc[0]/base_acc_out) ;
            #svm.SVC.


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
        #!!!!!!!!!!!!!!! calculate a constant weight 
        final_weight = [] ;
        final_hybrid_order = [] ; # top outlier should be the minimum
        weight_list = [] ;
        weight_standard = [0.1, 0.2, 0.3, 0.4] ;
        weight_standard_1 = [0.1, 1, 10 ,100] ;
        for p in permutations(weight_standard):
            weight_list.append(p) ;
        for t in permutations(weight_standard_1):
            weight_list.append(t) ;
        """
        hcv = Horizontal_cross_validation(data_name, train_x_copy, train_y_copy , weight_list, 5, 0.01) ; #(self, data_name, train_x_copy, train_y_copy , weight_list, block, base_size_percentage): # 
        hcv.main_control() ;
        final_weight = hcv.final_weight ;
        final_hybrid_order = hcv.make_final_hybrid_order(train_x_copy, train_y_copy, final_weight, base_size) ; # (self, train_x_copy, train_y_copy, final_weight, base_size):
        """
        #!!!!!!!!!!!!!!!
        for i in range (data_num_size) :
            hcv = Horizontal_cross_validation(data_name, train_x_copy, train_y_copy , weight_list, 5, 0.01) ; #(self, data_name, train_x_copy, train_y_copy , weight_list, block, base_size_percentage): # 
            hcv.main_control() ;
            final_weight = hcv.final_weight ;
            final_hybrid_order = hcv.make_final_hybrid_order(train_x_copy, train_y_copy, final_weight, base_size) ; # (self, train_x_copy, train_y_copy, final_weight, base_size):
            for j in range (base_size) :
                temp_pop_index = final_hybrid_order.index(min(final_hybrid_order)) ;
                final_hybrid_order.pop(temp_pop_index) ;
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

def cross_validation(data_name , pickle_path , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y): # use loop to put validation 
    number_of_method = 4 ;
    weight_standard = [0.1, 0.2, 0.3, 0.4] ;
    weight_standard_1 = [0.1, 1, 10 ,100] ;
    weight_list = [] ;
    method_order = [] ;
    angle_base_order = [] ;
    svm_outlier_order = [] ;
    rof_order = [] ;
    lof_order = [] ;
    hybrid_order = [] ;
    E_out = [] ;

    train_x_copy = copy.deepcopy(train_x) ;
    train_y_copy = copy.deepcopy(train_y) ;

    train_x_copy_o = copy.deepcopy(train_x) ;
    train_y_copy_o = copy.deepcopy(train_y) ;
    
    write_svm_file (train_x_copy , train_y_copy , data_name+'-temp3.train') ;
    cmd = '../../liblinear-incdec-2.01/train -s 2 -q ' + data_name+'-temp3.train' ;
    subprocess.call (cmd.split()) ;
    m = load_model (data_name+'-temp3.train.model') ;
    p_label_in , p_acc_in , p_val_in = predict(train_y , train_x , m) ;
    noise_acc_in = p_acc_in[0] ;
    p_label_out , p_acc_out , p_val_out = predict(test_y , test_x , m) ;
    noise_acc_out = p_acc_out[0] ;
    
    #E_in = [noise_acc_in/base_acc_in] ;
    #E_out = [noise_acc_out/base_acc_out] ;
    data_length = len(train_x) ;
    angle_base_order = Angle_Based(train_x_copy, train_y_copy, data_name) ;
    svm_outlier_order = SVM_Out(train_x_copy, train_y_copy, data_name) ;
    rof_order = Rof_Out(train_x_copy, train_y_copy, data_name, base_size) ;
    lof_order = Lof_Out(train_x_copy, train_y_copy, data_name) ;

    method_order.append(angle_base_order) ;
    method_order.append(svm_outlier_order) ;
    method_order.append(rof_order) ;
    method_order.append(lof_order) ;
    for p in permutations(weight_standard):
        weight_list.append(p) ;
    for t in permutations(weight_standard_1):
        weight_list.append(t) ;
    temp_weight = [0.25, 0.25, 0.25, 0.25] ;
    weight_list.append(temp_weight) ;
    hybrid_order = Make_hybrid_order(base_size, weight_list, method_order, data_length) ; # hybrid_order list stores order which calculate by all kinds of weight 
    #print("hybrid_order", hybrid_order) ;

    for i in range(len(weight_list)):
        train_x_pop = [] ;
        train_y_pop = [] ;
        temp_hybrid_order = [] ;
        temp_hybrid_instance = [] ;
        train_x_pop =  copy.deepcopy(train_x) ;
        train_y_pop = copy.deepcopy(train_y) ;
        temp_hybrid_order = copy.deepcopy(hybrid_order[i]) ;
        print("i", i) ;
        print("hybrid_order", len(hybrid_order[i])) ;
        print("length temp_hybrid_order", len(temp_hybrid_order)) ;
        print("base_size", base_size) ;
        for j in range(base_size): # pop points in different weight order
            pop_index = temp_hybrid_order.index(min(temp_hybrid_order)) ;
            temp_hybrid_order.pop(pop_index) ;
            train_x_pop.pop(pop_index) ;
            train_y_pop.pop(pop_index) ;
        write_svm_file (train_x_pop , train_y_pop , data_name+'-temp3.train') ;
        cmd = '../../liblinear-incdec-2.01/train -s 2 -q -i ' + data_name+'-temp3.train.model ' + data_name+'-temp3.train' ;
        subprocess.call (cmd.split()) ;
        m2 = load_model (data_name+'-temp3.train.model') ;
        # m2 = train(train_y_copy , train_x_copy , '-s 2 -q') ;
        p_label , p_acc , p_val = predict(test_y , test_x , m2) ;
        E_out.append(p_acc[0]) ;
    answer_index = E_out.index(max(E_out)) ;
    answer_weight = weight_list[answer_index] ;
    compare_index = 12 ;
    compare_weight = weight_list[compare_index] ;
    print("compare_index", compare_index) ;
    print("compare_weight", compare_weight) ;
    print("compare_acc", E_out[compare_index]) ;
    print("answer_index", answer_index) ;
    print("answer_weight", answer_weight) ;
    print("answer_acc", E_out[answer_index]) ;
    print("E_out", E_out) ;
    return answer_weight ;# answer_weight

                
            

    




def Make_hybrid_order(base_size, weight_list, method_order, data_length):
    hybrid_order = [] ;
    """
    for i in range(len(method_order)):
        print("i", i) ;
        print("length_method_order", len(method_order[i])) ;
        print("order")
        for j in range(30):
            print( method_order[i][j]) ;
    """
    for i in range(len(weight_list)):# len(weight_list)
        temp_weight_list = [] ;
        temp_weight_list = copy.deepcopy(weight_list[i]) ;
        print("i" , i) ;
        print("temp_weight_list", temp_weight_list) ;
        angle = 0.0 ;
        svm = 0.0 ;
        rof = 0.0 ;
        lof = 0.0 ;
        temp_hybrid_order = [] ; # record index which needs to be popped out
        for j in range(data_length):
            temp_hybrid_order.append(0.0) ;
        for j in range(data_length):
            for k in range(len(method_order)):
                temp_hybrid_order[j] = temp_hybrid_order[j] + temp_weight_list[k] * method_order[k].index(j) ;
        hybrid_order.append(temp_hybrid_order) ;
    return hybrid_order ;
def Angle_Based(train_x_copy_1, train_y_copy_1 ,data_name):
    train_x_copy = copy.deepcopy(train_x_copy_1) ;
    train_y_copy = copy.deepcopy(train_y_copy_1) ;
    index_order = [] ;
    instance_order = [] ;
    soa = Gen() ;
    soa.get_record_model(data_name) ;
    gen =  General(train_x_copy, soa.record_model) ;
    gen.main_control() ;
    parameter_k =5 ;
    svm_outlier_angle = Angle_Based_FastABOD(train_x_copy, gen.format_x, train_y_copy, parameter_k) ;
    svm_outlier_angle.main_control() ;
    for i in range(len(svm_outlier_angle.variance_cos)):
        svm_outlier_angle.variance_cos[i] = svm_outlier_angle.variance_cos[i] + 0.0000000001*i ;
    variance_cos = copy.deepcopy(svm_outlier_angle.variance_cos) ;
    data_length = len(svm_outlier_angle.variance_cos) ;

    for i in range(data_length):
        temp_index = svm_outlier_angle.variance_cos.index(min(svm_outlier_angle.variance_cos)) ;
        temp_instance = svm_outlier_angle.variance_cos.pop(temp_index) ;
        instance_order.append(temp_instance) ;

    for i in range(len(instance_order)):
        temp_instance = instance_order[i] ;
        index = variance_cos.index(temp_instance) ;
        index_order.append(index) ;
    return index_order ;
def SVM_Out(train_x_copy_1, train_y_copy_1 , data_name):
    train_x_copy = copy.deepcopy(train_x_copy_1) ;
    train_y_copy = copy.deepcopy(train_y_copy_1) ;
    index_order = [];
    instance_order = [] ;
    svo = Gen() ;
    svo.get_record_model(data_name) ;
    svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, svo.record_model) ;
    svm_outlier.main_control() ;

    for i in range(len(svm_outlier.compare_distance)):
        svm_outlier.compare_distance[i] = svm_outlier.compare_distance[i] +  0.0000000001*i ;
    compare_distance = copy.deepcopy(svm_outlier.compare_distance) ;
    data_length = len(svm_outlier.compare_distance) ;

    for i in range(data_length):
        temp_index = svm_outlier.compare_distance.index(min(svm_outlier.compare_distance)) ;# min
        temp_instance = svm_outlier.compare_distance.pop(temp_index) ;
        instance_order.append(temp_instance) ;

    for i in range(len(instance_order)):
        temp_instance = instance_order[i] ;
        index = compare_distance.index(temp_instance) ;
        index_order.append(index) ;
    return index_order ;
def Rof_Out(train_x_copy_1, train_y_copy_1, data_name, base_size):
    CR = [] ;
    train_x_copy = copy.deepcopy(train_x_copy_1) ;
    train_y_copy = copy.deepcopy(train_y_copy_1) ;
    index_order = [];
    instance_order = [] ;

    svr = Gen() ;
    svr.get_record_model(data_name) ;

    svm_outlier = SVM_Outlier(train_x_copy, train_y_copy, svr.record_model) ;
    svm_outlier.main_control() ;

    gen =  General(train_x_copy, svr.record_model) ;
    gen.main_control() ;

    rof = ROF_Method(gen.format_x, train_y_copy, base_size) ;
    rof.main_control() ;
    for i in range(len(rof.CR_List[len(rof.CR_List)-1].ROF)):
        rof.CR_List[len(rof.CR_List)-1].ROF[i] = rof.CR_List[len(rof.CR_List)-1].ROF[i] + 0.000000001*i; # for order convenience random.uniform(0.1,0.001)

    CR = copy.deepcopy(rof.CR_List[len(rof.CR_List)-1].ROF) ;
    data_length = len(rof.CR_List[len(rof.CR_List)-1].ROF) ;

    for i in range(data_length):
        temp_instance = min(rof.CR_List[len(rof.CR_List)-1].ROF) ;
        temp_index = rof.CR_List[len(rof.CR_List)-1].ROF.index(temp_instance) ;
        rof.CR_List[len(rof.CR_List)-1].ClusterSize.pop(temp_index) ;
        rof.CR_List[len(rof.CR_List)-1].label.pop(temp_index) ;
        rof.CR_List[len(rof.CR_List)-1].ROF.pop(temp_index) ;
        instance_order.append(temp_instance) ;

    for i in range(len(instance_order)):
        temp_instance = instance_order[i] ;
        index = CR.index(temp_instance) ;
        index_order.append(index) ;
        #!!!!!!!!!!!!!!!!!!!! WRONG 
    return index_order ;


def Lof_Out(train_x_copy_1, train_y_copy_1, data_name):
    train_x_copy = copy.deepcopy(train_x_copy_1) ;
    train_y_copy = copy.deepcopy(train_y_copy_1) ;
    index_order = [];
    instance_order = [] ;
    lof = Gen() ;
    lof.get_record_model(data_name) ;

    LOF_point = LOF(train_x_copy, train_y_copy, lof.record_model, K, MinPts) ; 
    LOF_point.main_control() ; 
    for i in range(len(LOF_point.all_LOF_list)):
        LOF_point.all_LOF_list[i] = LOF_point.all_LOF_list[i] + 0.00000001*i; # for order convenience
    lof_list = copy.deepcopy(LOF_point.all_LOF_list) ;

    for i in range(len(LOF_point.all_LOF_list)):
        temp_index = LOF_point.all_LOF_list.index(max(LOF_point.all_LOF_list)) ;#max
        temp_instance = LOF_point.all_LOF_list.pop(temp_index) ;
        instance_order.append(temp_instance) ;
    for i in range(len(instance_order)):
        temp_instance = instance_order[i] ;
        index = lof_list.index(temp_instance) ;
        index_order.append(index) ;
    return index_order ;

    

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
#pickle_name = " " ;
pickle_name = sys.argv[2] ;
#output_Folder = '../output/' ;
output_Folder = './' ;
pickle_path = output_Folder + pickle_name

test_file_name = train_file_name + '.t' ;
pure_data_name = (train_file_name.split('/'))[len(train_file_name.split('/'))-1] ;
data_num_size = 30 ;# originally 10

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

#per_wrong_label = percentage_wrong_label(pure_data_name, data_num_size,train_x , train_y) ;

select_method = 8 ;
E_in_8 , E_out_8 = run (pure_data_name , pickle_path ,select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Horizontal_cross_validation') ;
print (E_out_8) ;

"""
select_method = 7 ;
E_in_7 , E_out_7 = run (pure_data_name , pickle_path ,select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('ROF_UnLabel_method') ;
print (E_out_7) ;
"""

select_method = 1 ;
E_in_1 , E_out_1= run (pure_data_name, pickle_path , select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('random') ;
print (E_out_1) ;

"""
select_method = 6 ;
E_in_6 , E_out_6 = run (pure_data_name , pickle_path ,select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('Hybrid') ;
print (E_out_6) ;

select_method = 2 ;
E_in_2 , E_out_2 = run (pure_data_name , pickle_path ,select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('svm_outlier_angle') ;
print (E_out_2) ;

select_method = 3 ;
E_in_3 , E_out_3 = run (pure_data_name , pickle_path ,select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('svm_outlier_distance') ;
print (E_out_3) ;

select_method = 4 ;
E_in_4 , E_out_4 = run (pure_data_name , pickle_path ,select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('ROF') ;
print (E_out_4) ;

select_method = 5 ;
E_in_5 , E_out_5 = run (pure_data_name , pickle_path ,select_method , data_num_size , base_size , base_acc_in , base_acc_out , train_x , train_y , test_x , test_y) ;
print ('LOF') ; # Manhattan distance mean
print (E_out_5) ;
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
#query_num = np.arange(1.00 , 0.89 , -0.01) ; # (1.00 , 0.89 , -0.01)
#print("query_num", query_num) ;
#print("length of query_num", len(query_num)) ;
query_num = [] ;
for i in range(100, 99-data_num_size, -1):
    temp = i * 0.01 ;
    query_num.append(temp) ;
#query_num = np.arange(1.00 , 0.70 , -0.01) ; # (1.00 , 0.89 , -0.01)
plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4) ;
ax = plt.gca() ; 
ax.xaxis.set_major_locator( MultipleLocator(0.01) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
# ax.yaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('Acc rate') ;
plt.xlim(1.00 , 0.70) ; # (1.00 , 0.90)  (1.00 , 0.70)
# plt.ylim(0.8 , 1.2) ;
plt.grid () ;

plt.title(pure_data_name + '_in_' + ('%.3f' % base_acc_in)) ;
print("query_num") ;
print(query_num) ;
print("length of query_num", len(query_num)) ;
print("length of E_in_0", len(E_in_0)) ;
print("length of E_in_1", len(E_in_1)) ;
plt.plot(query_num, E_in_0, 'k', label='total') ;
plt.plot(query_num, E_in_1, 'bo--', label='random') ;
"""
plt.plot(query_num, E_in_2, 'rv--', label='SVM_Angle') ;
plt.plot(query_num, E_in_3, 'g^--', label='SVM_Dist') ;
plt.plot(query_num, E_in_4, 'c*--', label='ROF_NO_SVM') ;
plt.plot(query_num, E_in_5, 'mx--', label='LOF') ; #Man mean
plt.plot(query_num, E_in_6, 'yx--', label='hybrid') ;#Mah mean
"""
#plt.plot(query_num, E_in_7, 'rx--', label='ROF_UnLabel_method') ;
plt.plot(query_num, E_in_8, 'kx--', label='horizontal_cross') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_in_ROF_NO_SVM' + '.png') ;

plt.cla() ;

# ----------------------------------------------------------
query_num = [] ;
for i in range(100, 99-data_num_size, -1):
    temp = i * 0.01 ;
    query_num.append(temp) ;
#query_num = np.arange(1.00 , 0.69 , -0.01) ; # (1.00 , 0.89 , -0.01)
plt.subplot2grid((5,5), (0,0), colspan=5 , rowspan=4) ;
ax = plt.gca() ; 
ax.xaxis.set_major_locator( MultipleLocator(0.01) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
# ax.yaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('Acc rate') ;
plt.xlim(1.00 , 0.70) ; # (1.00 , 0.90) (1.00 , 0.70)
# plt.ylim(0.5 , 1.5) ;
plt.grid () ;

plt.title(pure_data_name + '_out' + ('%.3f' % base_acc_out)) ;

plt.plot(query_num, E_out_0, 'k', label='total') ;

plt.plot(query_num, E_out_1, 'bo--', label='random') ;
"""
plt.plot(query_num, E_out_2, 'rv--', label='SVM_Angle') ;
plt.plot(query_num, E_out_3, 'g^--', label='SVM_Dist') ;
plt.plot(query_num, E_out_4, 'c*--', label='ROF_NO_SVM') ;
plt.plot(query_num, E_out_5, 'mx--', label='LOF') ;# Man mean
plt.plot(query_num, E_out_6, 'yx--', label='hybrid') ; #Mah mean'
"""
#plt.plot(query_num, E_out_7, 'rx--', label='ROF_UnLabel_method') ;
plt.plot(query_num, E_out_8, 'kx--', label='horizontal_cross') ;

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3) ;
plt.savefig(pure_data_name + '_out_ROF_NO_SVM' + '.png') ;

plt.cla() ;

# ------------------------------------