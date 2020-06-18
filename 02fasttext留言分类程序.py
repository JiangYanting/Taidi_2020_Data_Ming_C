
#fasttext

import fasttext

#数据集格式，参考https://zhuanlan.zhihu.com/p/66739066
def train(): # 训练模型
    model = fasttext.train_supervised("train_new.txt", lr=0.1, dim=100,
             epoch=40, word_ngrams=2, loss='softmax')
    model.save_model("/model_file.bin")

def test(): # 预测
    classifier = fasttext.load_model("/model_file.bin") #加载保存的模型
    result = classifier.test("/test_new.txt")
    out = '/预测结果.txt'
    out1 = open(out,"w",encoding="utf8")

    print("准确率:", result)
    with open('/test_new.txt', "r",encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == '':
                continue
            out1.write(str(line)+" , "+str(classifier.predict([line])[0][0][0])+"\n")
    out1.close()
    
if __name__ == '__main__':
    train()
    test()
