import pickle ;

pk_file = open ('pendigits_LC_HW.pickle' , 'rb') ;
a = pickle.load(pk_file) ;
print (a[:10]) ;