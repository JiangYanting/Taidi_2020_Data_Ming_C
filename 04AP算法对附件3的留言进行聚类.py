
#对投票数大于5的留言聚类
from sklearn.cluster import AffinityPropagation
import numpy as np



#计算两个向量之间的余弦距离
def cosinesimilarity(vectorx, vectory):
    total = 0
    xsize = 0
    ysize = 0
    for i in range(0, len(vectorx)):
        total = total + float(vectorx[i])*float(vectory[i])
        xsize = xsize + float(vectorx[i])*float(vectorx[i])
        ysize = ysize + float(vectory[i])*float(vectory[i])
    xsize = math.sqrt(xsize)
    ysize = math.sqrt(ysize)
    if (ysize==0):
        return 0
    else:
        return total/xsize/ysize


#X = np.array([[1, 2], [1, 4], [1, 0],
#              [4, 2], [4, 4], [4, 0]])
#clustering = AffinityPropagation().fit(X)

#print(clustering.labels_)

#clustering.predict([[0, 0], [4, 4]])


file =r"附件3所有留言的向量.txt"

file1 = open(file,"r",encoding="utf8")

vec_all = []
num_all = []
for line in file1.readlines():
    line_list = line.strip().split("\t")
    vec_list = line_list[7].split(",")
    num = str(line_list[0])

    vec_all.append(vec_list)
    num_all.append(num)

vec_all_array = np.array(vec_all)
clustering = AffinityPropagation(preference=-50).fit(vec_all_array)
label_list = list(clustering.labels_)

out = r"留言聚类结果.txt" #聚类结果整理为 “03附件3加命名实体，经AP算法聚为701类后.txt”
out1 = open(out,"w",encoding="utf8")


for i,j in zip(num_all,label_list):
    #print(i,"\t",j)
    out1.write(str(i)+"\t"+str(j)+"\n")
    

    

