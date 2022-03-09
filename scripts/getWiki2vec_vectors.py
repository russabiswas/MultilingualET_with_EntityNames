import gensim
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec
import string


def readfilePandas(filename):
        df = pd.read_csv(filename, engine='python', header=None, names=['entity'])
        t = df.values.tolist()
        text = [item for sublist in t for item in sublist]
        return text

def readfiles(filename):
	with open(filename, 'r') as f:
		c = f.readlines()
	data = [x.strip() for x in c]
	return data


def loadmodel(wiki2vecfile):
        #coversion = glove2word2vec(embedding_file, wiki2vecfile)
        model = KeyedVectors.load_word2vec_format(wiki2vecfile)
        return model


def avgvectors(entityname,model):
        for c in string.punctuation:
                entityname = entityname.replace(c," ").lower()
                entityname = " ".join(entityname.split())
        tmp = []
        print('entity name--------')
        print(entityname)
        for i in entityname.split(' '):
                print(i)
                try:
                        tmp.append(model[i])
                        print(i)
                except:
                        print('inside except')
                        z= np.zeros((300,), dtype=int)
                        tmp.append(z)
        print('no of vectors for the word, ',entityname, 'is, ', len(tmp))
        vec_avg = np.mean(tmp, axis=0)
        return vec_avg


def getVectors(entities, wiki2vec_embed, op_vectorsFile):
	for i in entities:
		word = avgvectors(i.lower(),wiki2vec_embed)
		op_vectorsFile.write(i.replace(" ","_")+' '+ ' '.join([str(elem) for elem in word]))
		op_vectorsFile.write('\n')
	return 0



embed = loadmodel('eswiki_20180420_300d.txt')



#train vectors
train_entities = readfiles('entities')
trainfile = open('wiki2vec', 'w')
getVectors(train_entities, embed, trainfile)


