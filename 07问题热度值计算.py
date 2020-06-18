

#计算每个话题的热度值
import datetime
import time
import re
import xlrd
import math
score_dic = {}  #字典存储热度值，键为聚类号，值为热度
date_model = re.compile("(2019)"+"(\S){1,8}")


for c in range(701):
    file = r"04附件3保留下来的留言.txt"
    file1 = open(file,"r",encoding="utf8")

    gap_day = 0  #话题的间隔天数

    earliest = datetime.date(2050,12,31) #初始化该话题下的最早年月日
    latest = datetime.date(1950,1,1) #初始化该话题下的最晚年月日
    score = 0 # 初始化热度分数
    flag = 0 #标志该聚类号下有无留言，0为无，1为有
    
    for line in file1.readlines():
        line_list = line.strip().split("\t")
        clf = int(line_list[2])
        if clf == c:
            flag = 1
            date = str(line_list[4]).strip().split(" ")[0]  #获取时间
            date1 = re.sub("/","-",date)
            year = date1.split("-")[0]
            month = date1.split("-")[1]
            if month[0] == "0":
                month = month[1]

            day = date1.split("-")[2]
            if day[0] =="0":
                day = day[1]
            #print(year,"\t",month,"\t",day)
            y_m_d = datetime.date(int(year),int(month),int(day))
            if y_m_d > latest:
                latest = y_m_d
            if y_m_d < earliest:
                earliest = y_m_d

            vote = int(line_list[5]) + int(line_list[6])
            score = score + 10 + float(vote)/5

    if flag !=1 :
        continue
        
    gap_day = int((latest - earliest).days)
    #print(c,"\t",earliest,"\t",latest,"\t",gap_day)  #输出聚类话题编号、留言最早日期、最晚日期、间隔日期
    
    denominator = math.log(gap_day+5,2) + 1

    final_score = float(score) / denominator

    #print("话题：",c,"\t热度",final_score)

    score_dic[int(c)] = final_score



ordered_dic = sorted(score_dic.items(), key = lambda asd:asd[1], reverse = True)

for i in range(5):   
    print('类别=%s, 热度值=%s' % (ordered_dic[i][0],ordered_dic[i][1]))#输出前5类的热点问题类别，和对应的热度值
    
    
    
            
            
            
    
            
            
            
            
        
       
    
