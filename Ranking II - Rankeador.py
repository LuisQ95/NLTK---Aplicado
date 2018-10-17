#!/usr/bin/env python
# coding: utf-8

# # RANKING II - RANKEADOR

# In[3169]:


import numpy as np
import heapq
import re
import unicodedata
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
import ast
from sklearn.externals import joblib


# 
# De alguna forma, la pregunta llega a través de servicios y apoyada en el parámetro, usará el modelo que le corresponda.

# In[3170]:


1 #Si abrí mi Cuenta por la web, ¿Cómo puedo realizar operaciones en ella?.
2 #¿Qué es una Cuenta Sueldo bimoneda?.
3 #¿Puedo disponer de los fondos en cualquier momento?.
4 #¿Cuánto dinero necesito para abrir una Cuenta Sueldo?.
5 #¿Qué es una franquicia en mi cuenta Sueldo?.
6 #¿Mensualmente recibo estados de cuenta?.
7 #¿Puedo hacer nuevos depósitos en mi Cuenta Sueldo una vez abierta?.
8 #¿Cuántas operaciones libres tengo en Cajeros Automáticos del BBVA Continental?.
9 #¿Cuántas operaciones libres tengo por ventanilla?.
10#¿Cuánto pago al mes por mantenimiento de mi Cuenta Sueldo y mi Tarjeta de Débito?.
11#¿Los saldos de mi Cuenta Sueldo están cubiertos por el Fondo de Seguros de Depósito?.
12#¿Puedo usar mi tarjeta de débito en establecimientos comerciales que tengan terminales POS?.
#intento="¿Puedo disponer de los fondos en cualquier momento?"
#intento="¿quiero saber si es posible disponer de mis fondos en cualquier momento?"
#intento = "¿Puedo disponer de los fondos en cualquier momento?"
#intento = "¿Cuánto dinero necesito para abrir una Cuenta Sueldo?" 
intento = "¿Mensualmente recibo estados de cuenta?." 


# In[3171]:


parametro = "sueldo"

# Sí parámero == sueldo
clf =joblib.load("sueldo_model_.joblib")
clf.classes_


# #### 1. Proceso de conversión a minúsculas

# In[3172]:


intento = intento.lower()


# #### 2. Proceso de destrucción de tildes

# In[3173]:


def normalize(preguntas):
    return ''.join(unicodedata.normalize("NFD",i)[0] for i in str(preguntas))


# In[3174]:


intento = ''.join(normalize(c) for c in str(intento))


# #### 3. Destrucción de símbolos

# In[3175]:


intento_preg = [intento]
intento_preg
for i in range(len(intento_preg)):
    intento_preg[i] = re.sub("\W+", " ", intento_preg[i])
    intento_preg[i] = re.sub("[^\w]", " ", intento_preg[i])
    intento_preg[i] = re.sub('[ \t\n]+', ' ', intento_preg[i])
    intento_preg[i] = intento_preg[i].strip()


# #### 4. Stemming y eliminación de stopwords

# In[3176]:


stemmer = PorterStemmer()

for i in range(len(intento_preg)):
    palabras = nltk.word_tokenize(intento_preg[i])
    nuevaspalabras = [stemmer.stem(palabra) for palabra in intento_preg]
    intento_preg[i]=' '.join(nuevaspalabras)

intento_preg

my_list=stopwords.words('spanish')
my_list.append('si')
my_list.append('puedo')
my_list.append('cuanto')
my_list.append('cuanta')
my_list.append('estan')
my_list.append('hacer')
    
for i in range(len(intento_preg)):
    palabras = nltk.word_tokenize(intento_preg[i])
    nuevaspalabras = [palabra for palabra in palabras if palabra not in my_list]
 


# In[3177]:


nltk.word_tokenize(intento_preg[i])


# In[3178]:


[stemmer.stem(palabra) for palabra in palabras]


# In[3179]:


for i in range(len(intento_preg)):
    palabras = nltk.word_tokenize(intento_preg[i])
    nuevaspalabras = [stemmer.stem(palabra) for palabra in palabras]
    intento_preg[i]=' '.join(nuevaspalabras)


# In[3180]:


intento_preg


# #### 5. Creando el modelo TF-IDF  [PROBLEMA!!!: Necesito la lista freq_words]

# ##### 5.1 Invocando al objeto freq_words a partir del archivo freq_words.joblib

# In[3181]:


objects_sueldo =joblib.load("objects_sueldo.joblib")
freq_words = objects_sueldo[0]
freq_words


# In[3182]:


# IDF Dictionary

word_idfs_test = {}
for palabra in freq_words:
    doc_count = 0
    for pregunta in intento_preg:
        if palabra in nltk.word_tokenize(pregunta):
           doc_count += 1
    word_idfs_test[palabra] = abs(np.log(len(intento_preg)/(doc_count+1)))

word_idfs_test

# TF Matrix
    
tf_matrix_test = {}
for palabra in freq_words:
    doc_tf = []
    for pregunta in intento_preg:
        frequency = 0
        for w in nltk.word_tokenize(pregunta):
            if palabra == w:
               frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(pregunta))
        doc_tf.append(tf_word)
    tf_matrix_test[palabra] = doc_tf

tf_matrix_test

# Creating the Tf-Idf Model
    
tfidf_matrix_test = []
for palabra in tf_matrix_test.keys():
    tfidf_test = []
    for value in tf_matrix_test[palabra]:
        score = value * word_idfs_test[palabra]
        tfidf_test.append(score)
    tfidf_matrix_test.append(tfidf_test)   

tfidf_matrix_test
# Finishing the Tf-Tdf model
    
X_test = np.asarray(tfidf_matrix_test)
X_test = np.transpose(X_test)

###########

X_test=pd.DataFrame(X_test,columns=freq_words)


# In[3183]:


X_test


# In[3184]:


X_test["libr"]


# In[3185]:


word_idfs_test


# In[3186]:


predict_int=clf.predict(X_test)
predict_int


# In[3187]:


predict_probas=clf.predict_proba(X_test) 
predict_probas


# In[3188]:


predict_probas=np.concatenate(np.round(predict_probas,4))
predict_probas


# In[3189]:


order_probs=np.sort(predict_probas)
order_probs


# In[3190]:


order_probs.std()


# In[3191]:


prob_lim=order_probs[len(order_probs)-3]
prob_lim


# In[3192]:


pos_probs=np.where(predict_probas>=prob_lim)[0]
pos_probs


# In[3193]:


probs=predict_probas[pos_probs]
probs


# In[3194]:


df_probas=pd.DataFrame({'probas':probs,'posicion':pos_probs})
df_probas


# In[3195]:


df_probas=df_probas.sort_values(by='probas', ascending=False)
df_probas


# In[3196]:


pos_probs_ult=df_probas['posicion']
pos_probs_ult


# In[3197]:


pos_probs_ult.tolist()
np.asarray(pos_probs_ult)
pos_probs_ult=pos_probs_ult[:3]


# In[3198]:


pos_final=np.asarray(pos_probs_ult).tolist()
pos_final


# In[3199]:


intencion_a=np.array(clf.classes_)
intencion_a


# In[3200]:


pala_clave=intencion_a[pos_probs_1]
pala_clave


# In[3201]:


pala_clave_1=list(pala_clave)
pala_clave_1


# In[3202]:


probas=predict_probas[pos_final]
probas


# In[3203]:


orde=pd.DataFrame({'palabra':pala_clave_1,'probs':probas})
orde


# In[3204]:


orde_1=orde.sort_values(by='probs', ascending=False)
orde_1


# In[3205]:


etiquetas=list(orde_1['palabra'])
etiquetas


# ##### 5.2 Invocando al objeto descripcion_m a partir del archivo freq_words.joblib

# In[3206]:


descripcion_m = objects_sueldo[1]


# In[3207]:


descripcion_a=np.array(descripcion_m)


# In[3208]:


pala_clave_descrip=descripcion_a[pos_final]


# In[3209]:


descripcion_m


# In[3210]:


pala_clave_descrip


# In[3211]:


pala_clave_1_descrip=list(pala_clave_descrip)


# In[3212]:


orde_descrip=pd.DataFrame({'descripcion':pala_clave_1_descrip,'probs':probas})
orde_1_descrip=orde_descrip.sort_values(by='probs', ascending=False)
etiquetas_descrip=list(orde_1_descrip['descripcion'])


# In[3213]:


orde_1_descrip


# In[3214]:


etiquetas_descrip


# In[ ]:





# In[3215]:


flag=0
if len(np.unique(predict_probas))==1:
    flag=1
else:
    flag=0
flag



# ### Definición de hiperparámetros al modelo de Ranking: _umbral_

# El hiperparámetro tendrá como objetivo ajustar el modelo para que la respuesta sea única o tricotómica.
# 

# $\frac{1}{len(FAQ)}$ + ${umbral}$.

# In[3216]:


def zscore(probas):
    sd=probas.std()
    mean=probas.mean()
    zsc=(probas-mean)/sd
    return(zsc)


# In[3217]:


zscore(predict_probas)


# In[3218]:


import random
import scipy as sp
from scipy.stats import uniform


# In[3219]:


from math import factorial
from scipy.stats import norm
from sklearn.utils import resample
from scipy import stats


# In[3220]:


sd=predict_probas.std()
mean=predict_probas.mean()


# In[3221]:


stats.shapiro(predict_probas)
1/35


# In[3222]:


sd


# In[3223]:


mean


# In[3224]:


zsc=(predict_probas-mean)/sd
zsc # -2 a 2


# ## Probabilidades de cada pregunta del FAQ exacto
# 
# 1.-  3.17340639   0.40775613   
# 2.-  2.99460189   0.71222964   0.08093519   
# 3.-  3.2830604    0.12289531   
# 4.-  3.05485867   0.43962911   0.43962911   
# 5.-  2.99243639   0.69896325   0.10921301   
# 6.-  3.26887947   
# 7.-  2.98743358   0.38452115   0.38452115   0.32536405   
# 8.-  2.78763451   1.45940865   0.0327957    
# 9.-  3.14529559   0.6045702    0.14540296   
# 10.- 2.98872176   0.38983327   0.38983327   0.30320366   
# 11.- 2.98881752   0.38630975   0.30498138   0.30498138   
# 12.- 3.16990576   0.63757311   

# In[3225]:


np.where(zsc>=2)


# In[3226]:


1/len(descripcion_m)


# In[3227]:


sum(np.where(zsc>=2))


# In[3228]:


a = [1,3,2,2,2]
sum(predict_probas)


# In[3229]:


len(np.where(zsc>=2)[0])


# In[3230]:


if len(np.where(zsc>=2)[0])==0 or len(np.where(zsc>=2)[0])>=4:
    print("Por favor sea mas especifico")
elif len(np.where(zsc>=2)[0])>=1 or len(np.where(zsc>=2)[0])<=3:
    print("Estas son nuestras sugerencias")


# In[3231]:


n = len(predict_probas)
reps = 10000
xb = np.random.choice(x, (n, reps))
mb = xb.mean(axis=0)
mb.sort()

np.percentile(mb, [.05, .95])


# In[ ]:





# In[3232]:


c=1/35
c+0.1*c

