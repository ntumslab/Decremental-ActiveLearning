# python generate_badpoint.py input_file_name mode

# -----------------------------------------------

import sys ;
import random ;

# -----------------------------------------------

input_file_name = sys.argv[1] ;
mode = sys.argv[2] ;
output_file_name = input_file_name + '.n' + mode ;


# -----------------------------------------------

input_file = open (input_file_name , 'r') ;
output_file = open (output_file_name , 'w') ;

label_set = set() ;
feature_dict = dict() ;

sample_num = 0 ;
total_feature_num = 0 ;

while True :
	line = input_file.readline () ;
	if not line :
		break ;
	
	sample_num += 1 ;
	line_list = line.split() ;
	
	for i in range(len(line_list)) :
		if (i == 0) :
			label_set.add(line_list[0]) ;
		else :
			feature = line_list[i].split(':') ;
			feature[0] = int(feature[0]) ;
			if (feature[0] not in feature_dict) :
				feature_dict[feature[0]] = list() ;
			if (feature[1] not in feature_dict[feature[0]]) :
				feature_dict[feature[0]].append(feature[1]) ;
			total_feature_num += 1 ;
	
	output_file.write(line) ;	

input_file.close() ;
	
# -------------------------------------------------

noise_num = sample_num*2/100 ;
avg_feature_num = total_feature_num/sample_num ;

label_list = list(label_set) ;
feature_list = list(feature_dict) ;

label_num = len(label_list) ;

if (mode == '1') :
	for i in range (noise_num) :
		noise_label = label_list[random.randint(0 , label_num-1)] ;
		output_file.write(noise_label) ;
		
		random.shuffle(feature_list) ;
		temp_feature_list = sorted(feature_list[:avg_feature_num]) ;
		for j in range (avg_feature_num) :
			noise_feature_value = feature_dict[temp_feature_list[j]][random.randint(0,len(feature_dict[temp_feature_list[j]])-1)] ;
			output_file.write(' ' + str(temp_feature_list[j]) + ':' + noise_feature_value) ;
		
		output_file.write('\n') ;
elif (mode == '2') :
	input_file = open (input_file_name , 'r') ;
	while True :
		line = input_file.readline () ;
		if not line :
			break ;
			
		if (random.randint(1,100) <= 2) :
			line_list = line.split() ;
			noise_label = line_list[0] ;
			while (noise_label == line_list[0]) :
				noise_label = label_list[random.randint(0 , label_num-1)] ;
			output_file.write(noise_label) ;
			
			for feature in line_list[1:] :
				output_file.write(' ' + feature) ;
			
			output_file.write('\n') ;
output_file.close() ;