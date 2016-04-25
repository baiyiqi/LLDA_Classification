__author__ = 'yiqibai'
#encoding=utf-8

import numpy


class LLDA:
    def __init__(self, alpha, beta):
        """
        Initial function
        :param alpha: alpha
        :param beta: beta
        """
        self.alpha = alpha
        self.beta = beta



    def term_to_id(self, term):
        """
        Get the id of the word
        :param term: word
        :return: The id of the word
        """
        """
        self.vocas_id : {word, id}
        self.vocas : [w1, w2, ...]
        """
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id




    def complement_label(self, label):
        if not label:
            return numpy.ones(len(self.labelmap))
        vec = numpy.zeros(len(self.labelmap))
        vec[0] = 1.0
        for x in label:
            vec[self.labelmap[x]] = 1.0
        return vec




    def set_corpus(self, labelset, corpus, labels):
        """
        innite every necessary parameters in this class
        :param : from load_corpus()
        """
        """
        self.labels : [[]], 每行代表一篇文章，每列是一个label, 共有K列，如果文章在某label被标,则值为1，否则为0
        self.docs :  [[]], 每行代表一篇文章，每列是一个字, 共有len(self.vocas)列，如果字在文章出现,则值为1，否则为0
        """

        labelset.insert(0, "common")
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)

        self.vocas = []
        self.vocas_id = dict()
        self.labels = numpy.array([self.complement_label(label) for label in labels])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]

        M = len(corpus)   # 文章数
        V = len(self.vocas) # 词数

        self.z_m_n = []
        self.n_m_z = numpy.zeros((M, self.K), dtype=int) # doc-topic
        self.n_z_t = numpy.zeros((self.K, V), dtype=int) # topic-word
        self.n_z = numpy.zeros(self.K, dtype=int)

        for m, doc, label in zip(range(M), self.docs, self.labels):
            N_m = len(doc)
            '''
            z_n : 给doc的每个字赋label值，初始化阶段赋值0或者1, z_n 长度和doc的长度一样
            '''
            z_n = [numpy.random.multinomial(1, label / label.sum()).argmax() for x in range(N_m)]
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1   #doc_topic
                self.n_z_t[z, t] += 1   #topic_word
                self.n_z[z] += 1

        return self.vocas



    def inference(self):
        V = len(self.vocas)
        for m, doc, label in zip(range(len(self.docs)), self.docs, self.labels):
            for n in range(len(doc)):
                #t是文章第n个字的term id
                t = doc[n]
                # z 是 第m篇第n个字的label
                z = self.z_m_n[m][n]
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                denom_a = self.n_m_z[m].sum() + self.K * self.alpha
                denom_b = self.n_z_t.sum(axis=1) + V * self.beta
                p_z = label * (self.n_z_t[:, t] + self.beta) / denom_b * (self.n_m_z[m] + self.alpha) / denom_a
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1





    """topic-word distribution"""
    def phi(self):
        V = len(self.vocas)
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + V * self.beta)



    """document-topic distribution"""
    def theta(self):
        n_alpha = self.n_m_z + self.labels * self.alpha
        return n_alpha / n_alpha.sum(axis=1)[:, numpy.newaxis]



    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.phi()
        thetas = self.theta()

        log_per = N = 0
        for doc, theta in zip(docs, thetas):
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

