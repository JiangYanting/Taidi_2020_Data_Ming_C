import xlrd
import re
import hanlp
recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)


file =r"附件3.xlsx"
file1 = xlrd.open_workbook(file)
ret1 = file1.sheet_by_index(0)

out = r"附件3每条留言加上命名实体.txt" #命名实体识别的结果整理为“03附件3加命名实体，经AP算法聚为701类后.txt”
out1  = out1 = open(out,"w",encoding="utf8")

for i in range(1,4327):
    num = str(ret1.cell(i,0).value).strip()
    time =  str(ret1.cell(i,3).value).strip()

    text = str(ret1.cell(i,4).value).strip()
    text1= re.sub("\s","",text)
    
    topic = str(ret1.cell(i,2).value).strip()
    topic1 = re.sub("\s","",topic)

    for_vote = str(ret1.cell(i,6).value).strip()
    against_vote = str(ret1.cell(i,5).value).strip()

    string = ""

    for c in topic1:
        if c not in [" ","\t","\n"]:
            string = string + c
    for c in text1[:50]:
        if c not in [" ","\t","\n"]:
            string = string + c

    list_string = list(string)
    all_list = []
    all_list.append(list_string)
    a = recognizer(all_list)  #命名实体识别

    NE = ""
    for i in a[0]:  #每一个i是一个元组
        if not i:
            continue
        else:
            #print(i[0])
            NE = NE + i[0] + " "
    NE = NE.strip()
    out1.write(num+"\t"+topic1+"\t"+time+"\t"+text1+"\t"+against_vote+"\t"+for_vote+"\t"+NE+"\n")
    
    
    
out1.close()    


