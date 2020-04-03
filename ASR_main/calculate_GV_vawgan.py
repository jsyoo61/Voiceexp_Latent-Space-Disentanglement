import numpy as np
from scipy.spatial.distance import euclidean
import librosa
import os

from multiprocessing import Pool
import time
import math
import argparse
from dtw import dtw
from fastdtw import fastdtw
from preprocess_MCD import *
parser = argparse.ArgumentParser()
    
parser.add_argument('--model_iter', default='1')
args = parser.parse_args()
speakers=['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2']

source_dir = "./data/speakers_test/"
source_wavs = {}
source_wavs_path = {} #just for checking

mfcc_dim = 36 #36

test_dir = "./converted_voices/MCD_test/"

result_file_wav = "MCD_result_vaegan_wav.txt"
result_file_mfcc = "GV_result_vaegan_mfcc.txt"

counter = 0
for each_spk in speakers:
    for each_file in os.listdir(os.path.join(source_dir, each_spk)):
        if ".wav" in each_file:
            pure_file_name = each_file.split('.')[0] #del spk_id_sn
            source_file_key = each_spk + '_' + pure_file_name
            cur_wav_file_loc = os.path.join(source_dir, each_spk, each_file)
            cur_wav_data, _ = librosa.load(cur_wav_file_loc, sr = 16000, mono = True)
            source_wavs[source_file_key] = cur_wav_data
            source_wavs_path[source_file_key] = cur_wav_file_loc
            #print(cur_wav_file_loc)
            counter+=1

#print('total source file: {}'.format(counter))

#dist = lambda x,y: np.linalg.norm(x-y)
dist = lambda x,y: np.linalg.norm((x-y))


ori_whole_array=[]
pro_whole_array=[]
def calculateMCD_manual_mfcc(target_converted_file_path, compare_original_file_key, model_epoch, target_converted_source_id, compare_original_target_id):
    if target_converted_source_id == compare_original_target_id:
      return ""
    
    else:
    
      _,  temp = compare_original_file_key.split('_')
      pure_file_name =  temp
      
    
      target_conv_wav_data, sr = librosa.load(target_converted_file_path, sr = 22050, mono = True)
      
      _, _, _, _,sp_target = world_decompose(wav = target_conv_wav_data, fs = sr, frame_period = 5.0)
      #target_conv_wav_mfcc = world_encode_spectral_envelop(sp = sp_target, fs = sr, dim = mfcc_dim)
      target_conv_wav_mfcc = sp_target
      
      
      pro_whole_array.append(np.var(target_conv_wav_mfcc.T,axis=1).tolist())
      #ori_whole_array.append(np.var(compare_ori_wav_mfcc.T,axis=1).tolist())
      print(len(pro_whole_array))
      #print(len(ori_whole_array))
      
      #whole_array = np.asarray(whole_array)
      
        #whole_array= whole_array.tolist()
      sp_target = np.mean(np.var(target_conv_wav_mfcc.T,axis=1))
      
      #sp_com_ori = np.mean(np.var(compare_ori_wav_mfcc.T,axis=1))
      
      #distance, path = fastdtw(compare_ori_wav_mfcc[:,1:], target_conv_wav_mfcc[:,1:], radius = 1000000, dist = dist)
      #mcd = (10.0 / math.log(10)) * math.sqrt(2)* distance / len(path)
      resultLine = "epoch: " + model_epoch + " file: " + pure_file_name + ' ' + target_converted_source_id + " -> " + compare_original_target_id + " GV: " + str(sp_target)+"\n"
      print("{} {} {} {} {}".format(model_epoch, pure_file_name, target_converted_source_id, compare_original_target_id,str(sp_target)))
      return resultLine
    
    

def calculateMCD_mfcc(workData):
    #[target_converted_file_path, compare_original_file_key, model_epoch, target_converted_source_id, compare_original_target_id]
    target_converted_file_path = workData[0]
    compare_original_file_key = workData[1]
    model_epoch = workData[2]
    target_converted_source_id = workData[3]
    compare_original_target_id = workData[4]
    return calculateMCD_manual_mfcc(target_converted_file_path, compare_original_file_key, model_epoch, target_converted_source_id, compare_original_target_id)


workList = [] 
for each_dir_conv_way in os.listdir(test_dir):
        #print(each_dir_conv_way)
        _,cur_src, _, cur_trg = each_dir_conv_way.split('_') #extract src/trg info: [src]_to_[trg]
        for each_file in os.listdir(os.path.join(test_dir, each_dir_conv_way)):
            if ".wav" in each_file: #[spk_src]_to_[psk_trg]_[spk_id_sn]_[file_name].wav
               
                pure_file_name = each_file.split('.')[0]#del spk_id_sn
                #print(pure_file_name)
                cur_file_path = os.path.join(test_dir,  each_dir_conv_way, each_file)
                compare_original_file_key = cur_trg + '_' + pure_file_name # [target]_[file_name]
                
                workList.append([cur_file_path, compare_original_file_key, str(args.model_iter), cur_src, cur_trg])

#print("total works: {}".format(len(workList)))

with open(result_file_mfcc, 'a') as output:
    #p = Pool(1)
    output.writelines(map(calculateMCD_mfcc, workList))
    output.writelines("\n")
print("mfcc work done.")



pro_whole_array = np.average(pro_whole_array,axis=0)
np.savetxt('asr_proposed'+str(args.model_iter)+'.txt',pro_whole_array)

