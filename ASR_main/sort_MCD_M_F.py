import numpy as np
import os
#MCD_file = open("MCD_result_VCTK.txt",'r')
gender = open("gender.txt",'r')
#in_in_file = open("MCD_sort_M_M.txt",'a')
#in_out_file = open("MCD_sort_in_out.txt",'a')
#out_in_file = open("MCD_sort_out_in.txt",'a')
#out_out_file = open("MCD_sort_out_out.txt",'a')

gender_list = gender.readlines()
gender.close()
whole_gender_arr = []
for each_speaker in gender_list:
    whole_gender_arr.append(each_speaker)
print(whole_gender_arr)
#lines = MCD_file.readlines()
count=0
#for slt in sorted(os.listdir("processed")):
inout = ['in','out']
src_gender = str()
trg_gender = str()

result_MCD = open("MCD_result_vaegan_mfcc.txt",'r')
result = open("MCD_result_mean_std.txt",'a')


all_MCD_calc_M_F = []
all_MCD_calc_F_F = []
all_MCD_calc_F_M = []
all_MCD_calc_M_M = []
whole_MCD = []
lines = result_MCD.readlines()
#result.close()
#result_M_F = open("MCD_result_veagan_mfcc.txt",'a')
#result_F_F = open("MCD_result_veagan_mfcc.txt",'a')
#result_F_M = open("MCD_result_veagan_mfcc.txt",'a')
#result_M_M = open("MCD_result_veagan_mfcc.txt",'a')


for it in range(3):
  MCD_calc_M_F = []
  MCD_calc_F_F = []
  MCD_calc_F_M = []
  MCD_calc_M_M = []
  for line in lines:
          if line=="\n":
            pass
          else:
            _,iteration,_,file,src,_,trg,_,MCD=line.split(' ')
            if int(iteration)==it:
              for gd in whole_gender_arr:
                  #print(gd)
                  #idx = whole_gender_arr.index(gd)
                  if src in gd[:-1].split('  ')[0]:   
                      src_gender = gd[:-1].split('  ')[1]
                      print('source {} is {}'.format(src,src_gender))
                  if trg in gd[:-1].split('  ')[0]:   
                      trg_gender = gd[:-1].split('  ')[1]
                      print('target {} is {}'.format(trg,trg_gender))
              if src_gender=='M' and trg_gender=='M':
                  MCD_calc_M_M.append(float(MCD[:-1]))
                  all_MCD_calc_M_M.append(float(MCD[:-1]))
                  whole_MCD.append(float(MCD[:-1]))
              elif src_gender=='M' and trg_gender=='F':
                  MCD_calc_M_F.append(float(MCD[:-1]))
                  all_MCD_calc_M_F.append(float(MCD[:-1]))
                  whole_MCD.append(float(MCD[:-1]))
              elif src_gender=='F' and trg_gender=='F':
                  MCD_calc_F_F.append(float(MCD[:-1]))
                  all_MCD_calc_F_F.append(float(MCD[:-1]))
                  whole_MCD.append(float(MCD[:-1]))
              elif src_gender=='F' and trg_gender=='M':
                  MCD_calc_F_M.append(float(MCD[:-1]))
                  all_MCD_calc_F_M.append(float(MCD[:-1]))
                  whole_MCD.append(float(MCD[:-1]))
                  
              sub_file = open("MCD_sort_"+src_gender+"_to_"+trg_gender+".txt",'a')
              sub_file.write(line)
              sub_file.close()           
  
  MCD_calc_M_F = np.asarray(MCD_calc_M_F)
  MCD_calc_M_M = np.asarray(MCD_calc_M_M)
  MCD_calc_F_M = np.asarray(MCD_calc_F_M)
  MCD_calc_F_F = np.asarray(MCD_calc_F_F)
  result_F_F =  "Model" + str(it) + "FtoF" + " " + str(np.mean(MCD_calc_F_F)) + " " +  str(np.std(MCD_calc_F_F)) +"\n"
  result_M_F =  "Model" + str(it) + "MtoF" + " " + str(np.mean(MCD_calc_M_F)) + " " +  str(np.std(MCD_calc_M_F)) +"\n"
  result_F_M =  "Model" + str(it) + "FtoM" + " " + str(np.mean(MCD_calc_F_M)) + " " +  str(np.std(MCD_calc_F_M)) +"\n"
  result_M_M =  "Model" + str(it) + "MtoM" + " " + str(np.mean(MCD_calc_M_M)) + " " +  str(np.std(MCD_calc_M_M)) +"\n"
  
  result.write(result_F_F)
  result.write(result_M_F)
  result.write(result_F_M)
  result.write(result_M_M)


  
  #all_MCD_calc_F_F.append(np.mean(MCD_calc_F_F))
  #all_MCD_calc_M_F.append(np.mean(MCD_calc_M_F))
  #all_MCD_calc_F_M.append(np.mean(MCD_calc_F_M))
  #all_MCD_calc_M_M.append(np.mean(MCD_calc_M_M))

all_MCD_calc_F_F = np.asarray(all_MCD_calc_F_F)
all_MCD_calc_M_F = np.asarray(all_MCD_calc_M_F)
all_MCD_calc_F_M = np.asarray(all_MCD_calc_F_M)
all_MCD_calc_M_M = np.asarray(all_MCD_calc_M_M)

result_F_F =  "FtoF" + " " + str(np.mean(all_MCD_calc_F_F)) + " " +  str(np.std(all_MCD_calc_F_F)) +"\n"
result_M_F =  "MtoF" + " " + str(np.mean(all_MCD_calc_M_F)) + " " +  str(np.std(all_MCD_calc_M_F)) +"\n"
result_F_M =  "FtoM" + " " + str(np.mean(all_MCD_calc_F_M)) + " " +  str(np.std(all_MCD_calc_F_M)) +"\n"
result_M_M =  "MtoM" + " " + str(np.mean(all_MCD_calc_M_M)) + " " +  str(np.std(all_MCD_calc_M_M)) +"\n"

result_whole = "Average" + " " + str(np.mean(whole_MCD)) + " " +  str(np.std(whole_MCD)) +"\n"


result.write(result_F_F)
result.write(result_M_F)
result.write(result_F_M)
result.write(result_M_M)
result.write(result_whole)


  

  
result.close()