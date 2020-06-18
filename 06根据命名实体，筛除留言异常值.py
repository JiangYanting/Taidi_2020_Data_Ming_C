
import pynlpir
pynlpir.open()


from collections import Counter
def counter_most_3(arr):
    return Counter(arr).most_common(3) #返回一个列表里最高频的3个元素


#根据命名实体的分布情况，初步筛除非热点的留言与话题

file= r"03附件3加命名实体，经AP算法聚为701类后.txt"
file1 = open(file,"r",encoding="utf8")
Not_NE_note = 0


topic_NE = {}  #键为聚类号，值为一个列表，存储的是该聚类号下所有的命名实体（可以重复）

#第一部分：得到每个聚类话题下的命名实体和细粒度分词后的片段
for i in range(701):
    topic_NE[i] = []  

    for line in file1.readlines():
        line_list = line.strip().split("\t")
        if int(line_list[2]) == i:
            if len(line_list) < 8:
                Not_NE_note = Not_NE_note + 1
                continue
            else:
                #print("有命名实体")
                NE_list = line_list[7].split(" ")
                for NE in NE_list:
                    if NE in ["3","A","A3","7","A7","A5","楼","路","5","6","市","A3区","A4","A4区"]: #排除部分识别错误的命名实体
                        continue
                    else:
                        topic_NE[i].append(NE)
                    segments=pynlpir.segment(NE)  #对命名实体再细粒度分词后，得到的词语列表
                    for x in range(len(segments)):
                        segment = segments[x]
                        word=str(segment[0])
                        if str(word) != str(NE):
                            if (len(word)>1) and (word not in ["小区","A3区","A4","A4区","A3","A7","A5","A2","A4","A6"]):
                                topic_NE[i].append(word)
                #print(topic_NE[i])
    file1.close()
    file1 = open(file,"r",encoding="utf8")
                        
                    
print(len(topic_NE),"topic_NE的长度")                
print("不含命名实体的留言数量：",Not_NE_note)

outfile = r"04附件3保留下来的留言"
outfile1 = open(outfile,"w",encoding="utf8")

outfile2 = r"05附件3被筛除的留言"
outfile3 = open(outfile2,"w",encoding="utf8")

'''
1.没有命名实体的留言，筛除
2.所拥有的命名实体，在该话题下的命名实体列表中都只出现了一次的，筛除。
3.不含该话题最高频的3个命名实体的留言，筛除。
'''


for i in range(701):
    topic_i_NE = topic_NE[i]
    most_3 = []  #得到一个话题下出现次数最多的3个命名实体
    for item in counter_most_3(topic_i_NE): 
        most_3.append(item[0])
    if i == 254:
        print(most_3)
        
    for line in file1.readlines():
        line_list = line.strip().split("\t")
        if int(line_list[2]) == i:
            if len(line_list) < 8:
                outfile3.write(line)
                continue    # 1.没有命名实体的留言，筛除
            else:
                flag = 0 # 根据上述条件2、条件3，如果flag变为1，就不用被筛除
                NE_list = line_list[7].split(" ")
                for NE in NE_list:
                    if topic_i_NE.count(NE) > 1:
                        flag = flag + 1
                        #print("嘻嘻嘻")
                        break

                for NE in NE_list:
                    if NE in most_3:
                        #print("哈哈哈")
                        flag = flag + 1
                        break
                        
                if flag == 2:
                    outfile1.write(line)  #outfile1，保留的
                elif flag < 2 :
                    outfile3.write(line)  #outfile3，筛除的
    file1.close()
    file1 = open(file,"r",encoding="utf8")


                
                    
                
                
                
            
                
                
                
            
    
        
        
    
