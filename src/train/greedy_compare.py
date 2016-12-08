import sys ;
import pickle ;
import numpy as np ;

import matplotlib ;
matplotlib.use('Agg') ;
import matplotlib.pyplot as plt ;
from matplotlib.ticker import MultipleLocator, FuncFormatter ;

train_file_name_origin = sys.argv[1] ;
pure_data_name = (train_file_name_origin.split('/'))[len(train_file_name_origin.split('/'))-1] ;

gr_file = open ('../../data/data_greedy/' + pure_data_name + '_99per_Eout.pickle' , 'rb') ;
_ , greedy_pop_list , __ , ___ = pickle.load(gr_file) ;
gr_file.close() ;

lc_file = open ('../../data/LC_HW/' + pure_data_name + '_LC_HW.pickle' , 'rb') ;
lc_pop_list = pickle.load(lc_file) ;
lc_file.close() ;

min_len = min(len(greedy_pop_list) , len(lc_pop_list)) ;
temp_set = set() ;
same_set_list = [] ;
best_set_list = [] ;
rank_list = [] ;
for i in range (min_len) :
	temp_set.add(lc_pop_list[i]) ;
	temp_set.add(greedy_pop_list[i]) ;
	temp_num = (i+1)*2 - len(temp_set) ;
	same_set_list.append(temp_num) ;
	best_set_list.append(i+1) ;
	if (lc_pop_list[i] in greedy_pop_list) :
		rank_list.append (greedy_pop_list.index(lc_pop_list[i]) + 1) ;
	else :
		rank_list.append (1000) ;
	
query_num = np.arange(1.00 , 0.01 , -0.001) ;
ax = plt.gca() ; 
ax.xaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
ax.yaxis.set_major_locator( MultipleLocator(100) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('sample num') ;
plt.xlim(1.00 , 0.00) ;
plt.ylim(0 , 1000) ;
plt.grid () ;

plt.title(pure_data_name + '_Gr_LCHW') ;
plt.plot(query_num, same_set_list , 'b') ;
plt.plot(query_num, best_set_list , 'grey') ;
plt.savefig(pure_data_name + '_compare' + '.png') ;

plt.cla() ;

query_num = np.arange(1.00 , 0.01 , -0.001) ;
ax = plt.gca() ; 
ax.xaxis.set_major_locator( MultipleLocator(0.1) ) ;
# ax.xaxis.set_minor_locator( MultipleLocator(0.01) ) ;
ax.yaxis.set_major_locator( MultipleLocator(100) ) ;
# ax.yaxis.set_minor_locator( MultipleLocator(0.02) ) ;
plt.xlabel('% of Data') ;
plt.ylabel('rank') ;
plt.xlim(1.00 , 0.90) ;
plt.ylim(0 , 1000) ;
plt.grid () ;

plt.title(pure_data_name + '_Gr_LCHW') ;
plt.plot(query_num, rank_list , 'r') ;
plt.plot(query_num, best_set_list , 'grey') ;
plt.savefig(pure_data_name + '_compare_2' + '.png') ;

plt.cla() ;