__author__ = 'yiqibai'
#encoding=utf-8



import  re
import math
import jieba
import pymongo




IP = "129.63.16.167"    # ip mongodb
PORT = 27017            # port mongodb
DB = "KuaiWenCategory"  # db mongodb
COLLECTION = "APP"      # collection mongodb





'''load stop words'''
f = open("stopword.txt", 'r')
stopword = list()
for l in f.readlines():
    stopword.append(l.replace('\r\n','').decode("utf-8"))






'''segment'''
def segmet(text):
    '''
    :param :    text
    :return :   segment text separated with whitespace
    '''
    tre = ur'[\u4e00-\u9fff]+'
    utext = ''.join(re.findall(tre, text))
    segText = [w for w in filter(lambda x : len(x) > 1 and x not in stopword, jieba.cut(utext))]
    # return ' '.join(segText)
    return segText






'''get phi from db'''
def getPhi():
    '''
    :return : {category : T, {keyword : { W, distribution}}}
    '''
    try:
        con = pymongo.MongoClient(IP, PORT) #get data from tencent server
        db = con[DB]                                                                #
        cur = db[COLLECTION]
        Phi = {}
        for i in cur.find():
            Phi = i
        del Phi[u"_id"]
    except Exception, e:
        print e.message
    con.close()
    return Phi






Phi_data = getPhi()
'''using probablistic'''
def classify1(text):
    '''
    :param :   text
    :return :   label
    '''
    segText = segmet(text.decode("utf-8"))
    cates = dict()
    for c in Phi_data:
        coKey = set(segText) & set(Phi_data[c].keys())
        cates[c] = 0.0
        for k in coKey:
            cates[c] += -math.log(Phi_data[c][k])
    scates = sorted(cates.items(), key=lambda d : d[1], reverse=True) #descending
    return scates[0][0]





'''use word match'''
def classify2(text):
    '''
    :param :   text
    :return :   label
    '''
    segText = segmet(text.decode("utf-8"))
    cates = dict()
    for c in Phi_data:
        cates[c] = len(set(segText) & set(Phi_data[c].keys()))
    scates = sorted(cates.items(), key=lambda d : d[1], reverse=True) #descending
    return scates[0][0]


# text = "对此，有律师指出，作为第三方平台，瓜子二手车直卖网或其他的二手车网站，应承担销售中的连带责任。如今，通过互联网平台交易二手车，已经是许多车主的选择。一系列的案例告诉我们，网上二手车买卖，依然问题多多，并且，一旦交易成功，售后很难得到保障。"
# print classify1(text)
# print classify2(text)