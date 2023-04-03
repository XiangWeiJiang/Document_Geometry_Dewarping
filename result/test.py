import fastwer
import distance
import pytesseract
from PIL import Image

with open("ocr_files.txt","r") as f:
    name = f.readlines()
print(len(name))

import json
f = open('test_gt.json', 'r')

content = f.read()
b = json.loads(content)

CER = []
WER = []
ED = []  
for fn in name:
    fn =  fn[:-1]
    for j in range(2): 
        
        fm = fn.split("/")[-1]+"_"+str(j+1)
        img = Image.open("grid/"+fm+" copy.png")  
        data = pytesseract.image_to_string(img,config='--psm 6')
        pred = data#
        gt = b[fn]
 
        # Obtain Sentence-Level Character Error Rate (CER)
        cer = fastwer.score_sent( pred,gt,char_level=True)
        # Obtain Sentence-Level Word Error Rate (WER)
        wer = fastwer.score_sent(pred,gt )
        ed = distance.levenshtein(pred,gt)

        CER.append(cer)
        WER.append(wer)
        ED.append(ed)
        print(cer,wer,ed)

import numpy as np
print("cer_mean",len(CER),np.mean(CER),np.std(CER))
print("wer_mean",len(WER),np.mean(WER),np.std(WER))
print("ed_mean",len(ED),np.mean(ED),np.std(ED))
