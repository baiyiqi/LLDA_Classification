__author__ = 'yiqibai'
#encoding=utf-8



import pymongo
import  re, numpy
import LLDAInference
from optparse import OptionParser



IP = "129.63.16.167" #ip for save data
PORT = 27017         #port for save data
DB = "KuaiWenCategory"  #db for save data
COLLECTION = "APP"      #defined by each application
TPNUM = 100             #top numbers of keywords under each category to save in db




'''load training corpus'''
def load_traincorpus(fname):
    '''
    :param :    filename (eg: total200)
    :return :   corpus = [d1. d2, ..], di = [w1, w2, ...]
                labels = [dl1, dl2, ...], dli = [l1,..] dli是每篇文章的label,对应corpus的每个doc
                labelmap 所有label
    '''
    corpus = []
    labels = []
    labelmap = dict()
    f = open(fname, 'r')
    for line in f:
        mt = re.match(r'\[(.+?)\](.+)', line)
        if mt:
            label = mt.group(1).split(',') #extract lables
            label1 = []
            for x in label:
                #x = x.decode("utf-8")
                labelmap[x] = 1
                label1.append(x)
            txt = mt.group(2) #extract wblog
        else:
            label1 = None
        doc = txt.split(" ")
        if len(doc) > 0:
            corpus.append(doc)
            labels.append(label1)
    f.close()

    return labelmap.keys(), corpus, labels




'''save phi to db'''
def save2mongo(phiData):
    '''
    :param : {category : T, {keyword : { W, distribution}}}
    '''
    try:
        con = pymongo.MongoClient(IP, PORT) #get data from tencent server
        dn = DB
        cn = COLLECTION
        db = con[dn]                                                                #
        cur = db[cn]
        cur.save(phiData)
    except Exception, e:
        print e.message
    con.close()




'''main file  alpha = 50 / T, beta = 0.1'''
def TrainLLDA(trainfile):
    '''
    :param : trainfile (eg: total200)
    '''

    lbfilter = "common"
    #load training data
    labelset, corpus, labels = load_traincorpus(trainfile)
    #set parameters
    K = len(labelset)
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default = 50.0/ K)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default = 0.1)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default = 64) #iteration
    (options, args) = parser.parse_args()

    #training
    print 'training starts'
    llda = LLDAInference.LLDA(options.alpha, options.beta)
    vocabulary = llda.set_corpus(labelset, corpus, labels)
    for i in range(options.iteration):
        print i, '\r',
        llda.inference()

    #convert phi to phidict
    phi = llda.phi()
    T_W_Dist = dict()
    for k, label in enumerate(labelset):
        if label == lbfilter:
            continue
        print "\n-- label %d : %s" % (k, label)
        T_W_Dist[label] = dict()
        for w in numpy.argsort(-phi[k])[:TPNUM]: #get TPNUM keywords under each category
            print "%s: %.4f" % (llda.vocas[w], phi[k,w])
            T_W_Dist[label][llda.vocas[w]] = phi[k,w]


    #save phi to database
    save2mongo(T_W_Dist)
    #save phi to file
    numpy.savetxt("PHI", phi, delimiter=",")



TrainLLDA("total200")
