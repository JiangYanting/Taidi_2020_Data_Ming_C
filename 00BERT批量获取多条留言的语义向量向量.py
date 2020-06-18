import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertConfig
#import matplotlib.pyplot as plt
import numpy as np
import os
import re
#import matplotlib
#from sklearn.decomposition import PCA

# OPTIONAL: if you want to have more information on what's happening
#under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
"加载预训练模型的分词器、词汇表"
tokenizer = BertTokenizer.from_pretrained('/bert-base-chinese')#BERT中文模型的路径
#模型下载地址https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
    


# Load pre-trained model (weights)
#'加载预训练模型'
model = BertModel.from_pretrained('/bert-base-chinese')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
#'''模型设置成评估模式，去除dropout（随机停止更新部分参数）的模块
#因为在评估时，有可重复的结果是很重要的'''
model.eval()


import xlrd
import re

file = r"附件3每条留言加上命名实体.xlsx"
file1 = xlrd.open_workbook(file)

ret1 = file1.sheet_by_index(0)

out= r"附件3所有留言的向量.txt"
out1 = open(out,"w",encoding="utf8")

for j in range(1,4327):
    num = str(ret1.cell(j,0).value).strip()
    topic = str(ret1.cell(j,1).value).strip()
    time = str(ret1.cell(j,2).value).strip()
    text = str(ret1.cell(j,3).value).strip()
    text1 = re.sub("\s","",text)
    
    for_vote = str(ret1.cell(j,5).value).strip()
    against_vote = str(ret1.cell(j,4).value).strip()
    NE = str(ret1.cell(j,6).value).strip()

    string = ""

    for i in topic:
        if i not in [" ","\n","\t"]:
            string = string +i + " "    
    sent1 =string.strip()

    
    sent = "[CLS] " + sent1 + " [SEP]"
    tokenized_text = tokenizer.tokenize(sent)
    #print(tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) # Map the token strings to their vocabulary indeces.
    
    segments_ids = [1] * len(tokenized_text)

    #print (segments_ids)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        # 当网络中的某一个tensor不需要梯度时，可以使用torch.no_grad()来处理。
        # See the models docstrings for the detail of the inputs
        try:
            outputs = model(tokens_tensor, segments_tensors)
            encoded_layers = outputs[0] # 第一个元素，是bert模型的最后一层的隐藏状态。
            # print(encoded_layers.size())
            token_vecs = encoded_layers[0]   #得到一句话里每个词的768维向量的矩阵。
            #print(token_vecs.size())
            sentence_embedding = torch.mean(token_vecs, dim=0)
            #print(sentence_embedding.size())
            b = sentence_embedding.numpy().tolist()
        except:
            print("提取向量有误")
            continue
        
    vec_string = ""

    for k in b:
        vec_string = vec_string + str(k)+","
    vec_string = vec_string.strip(",")
    out1.write(str(num)+"\t"+topic+"\t"+time+"\t"+text1+"\t"+against_vote+"\t"+for_vote+"\t"+NE+"\t"+vec_string+"\n")  #line1的倒数第二个数，就是类别标签
    

print("done!")    

out1.close()



