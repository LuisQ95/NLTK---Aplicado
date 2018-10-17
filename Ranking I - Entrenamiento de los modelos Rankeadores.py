#!/usr/bin/env python
# coding: utf-8

# # RANKING I - ENTRENAMIENTO DE LOS MODELOS RANKEADORES

# De alguna forma, esta data debe ser traída de la base de conocimientos, cuyo producto (parámetro) sea "Sueldo". Para esto, se apoyará en lo que venga desde el servicio "reformulador"

# In[115]:


FAQ=["Si abrí mi Cuenta por la web, ¿Cómo puedo realizar operaciones en ella?.",
            "¿Qué es una Cuenta Sueldo bimoneda?.",
            "¿Cuánto pago al mes por mantenimiento de mi Cuenta Sueldo y mi Tarjeta de Débito?.",
            "¿Cuántas operaciones libres tengo por ventanilla?.",
            "¿Cuántas operaciones libres tengo en Cajeros Automáticos del BBVA Continental?.",
            "¿Puedo usar mi tarjeta de débito en establecimientos comerciales que tengan terminales POS?.",
            "¿Puedo disponer de los fondos en cualquier momento?.",
            "¿Los saldos de mi Cuenta Sueldo están cubiertos por el Fondo de Seguros de Depósito?.",
            "¿Cuánto dinero necesito para abrir una Cuenta Sueldo?.",
            "¿Puedo hacer nuevos depósitos en mi Cuenta Sueldo una vez abierta?.",
            "¿Qué es una franquicia en mi cuenta Sueldo?.",
            "¿Mensualmente recibo estados de cuenta?.",    
            "¿Qué es Twitter?",
            "¿Necesito algo en especial para usar el servicio?",
            "¿Qué es un Tweet?",
            "¿Cómo puedo enviar actualizaciones a Twitter?",
            "¿Qué es un Retweet?",
            "¿Cómo publico una imagen en Twitter?",
            "¿Puedo editar un Tweet una vez que lo publiqué?",
            "¿Cómo activo el modo nocturno?",
            "¿Quiénes leen mis actualizaciones?",
            "¿Por qué no puedo ver todos mis Tweets? ¿Se perdieron?",
            "¿Puedo mostrar mis actualizaciones de Twitter en mi blog?",
            "¿Qué significa seguir a alguien en Twitter?",
            "¿Cómo puedo encontrar personas para seguir?",
            "¿Cómo sé a quién estoy siguiendo?",
            "¿Cómo sé quién me sigue?",
            "¿Qué son los límites de seguimiento?",
            "¿Qué son las respuestas?",
            "¿Cuál es la diferencia entre una respuesta y un Mensaje Directo?",
            "¿Qué son los Mensajes Directos?",
            "¿Por qué se suspenden las cuentas?",
            "¿Cómo puedo denunciar el spam?",
            "¿Dónde puedo encontrar más información sobre los Términos de servicio de Twitter?",
            "¿Cómo puedo enviar una queja sobre problemas relacionados con los derechos de autor, la suplantación de identidad, las marcas comerciales u otras cuestiones relacionadas con los Términos de servicio?"]


# In[116]:


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


# #### 1. Proceso de conversión a minúsculas

# In[117]:


descripcion = [item.lower() for item in FAQ]
preguntas = descripcion


# #### 2. Proceso de destrucción de tildes

# In[118]:


def normalize(preguntas):
    return ''.join(unicodedata.normalize("NFD",i)[0] for i in str(preguntas))
preguntas = ast.literal_eval(normalize(preguntas))
#Resultado: array donde cada elemento es una pregunta sin tildes y en minúscula


# #### 3. Destrucción de símbolos

# In[119]:


for i in range(len(preguntas)):
    preguntas[i] = re.sub("\W+", " ", preguntas[i])
    preguntas[i] = re.sub("[^\w]", " ", preguntas[i])
    preguntas[i] = re.sub('[ \t\n]+', ' ', preguntas[i])
    preguntas[i] = preguntas[i].strip()
#Resultado: array donde cada elemento es una pregunta sin tildes, en minúscula y sin signos de interrogación


# #### 4. Stemming y eliminación de stopwords

# In[120]:


stemmer = PorterStemmer()
my_list=stopwords.words('spanish')
my_list.append('si')
my_list.append('puedo')
my_list.append('cuanto')
my_list.append('cuanta')
my_list.append('estan')
my_list.append('hacer')

#Stemmizando
for i in range(len(preguntas)):
    palabras = nltk.word_tokenize(preguntas[i])
    nuevaspalabras = [stemmer.stem(palabra) for palabra in palabras]
    preguntas[i]=' '.join(nuevaspalabras)

#Quitando stopwords
for i in range(len(preguntas)):
    palabras = nltk.word_tokenize(preguntas[i])
    nuevaspalabras = [stemmer.stem(palabra) for palabra in palabras]
    nuevaspalabras = [palabra for palabra in palabras if palabra not in my_list]
    preguntas[i]=' '.join(nuevaspalabras)


# In[121]:


preguntas


# #### 5. Creación de las "autoetiquetas" (target improvisado)

# In[122]:


def remove_char(s):
        a=s[0]
        b=s[len(s)-1]
        nuevaletra=a+b
        return nuevaletra

intencion = ["" for x in range(len(preguntas))]
for i in range(len(preguntas)):
    palabras = nltk.word_tokenize(preguntas[i])
    strs = ["" for x in range(len(palabras))]
    for j in range(len(palabras)):
        strs[j]=remove_char(palabras[j])
    intencion[i]=''.join(strs)


# In[123]:


intencion


# In[124]:


df_3=pd.DataFrame({'Descripcion':FAQ,'Intencion':intencion})
df_3=df_3.sort_values(by='Intencion', ascending=True)
descripcion_m=df_3['Descripcion']
descripcion_m=descripcion_m.tolist()


# #### 6. Creamos el modelo TF-IDF

# In[125]:


word2count = {}
for pregunta in preguntas:
    palabras = nltk.word_tokenize(pregunta)
    for palabra in palabras:
        if palabra not in word2count.keys():
           word2count[palabra] = 1
        else:
           word2count[palabra] += 1

freq_words = heapq.nlargest(100000, word2count,key=word2count.get)

# Diccionario IDF

word_idfs = {}
for palabra in freq_words:
    doc_count = 0
    for pregunta in preguntas:
        if palabra in nltk.word_tokenize(pregunta):
           doc_count += 1
    word_idfs[palabra] = np.log(len(preguntas)/(1+doc_count))

word_idfs

# Matriz TF
    
tf_matrix = {}
for palabra in freq_words:
    doc_tf = []
    for pregunta in preguntas:
        frequency = 0
        for w in nltk.word_tokenize(pregunta):
            if palabra == w:
               frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(pregunta))
        doc_tf.append(tf_word)
    tf_matrix[palabra] = doc_tf

tf_matrix

# Modelo tf-Idf

tfidf_matrix = []
for palabra in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[palabra]:
        score = value * word_idfs[palabra]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)   
    
tfidf_matrix    

# Finalizando el Tf-Tdf model
    
X = np.asarray(tfidf_matrix)
X = np.transpose(X)

###########
X_t=pd.DataFrame(X,columns=freq_words)

X = pd.DataFrame(X,columns=freq_words)
Y = pd.DataFrame(intencion,columns=['Intencion'])


# In[141]:


word2count


# In[126]:


clf = MultinomialNB()
clf.fit(X=X, y=Y)


# In[127]:


clf.classes_


# In[128]:


joblib.dump(clf, 'sueldo_model_.joblib') 


# #### 7. Exportando las palabras frecuentes (objeto que se utilizará frecuentemente en cada predicción)

# In[129]:


joblib.dump(freq_words, 'freq_words_sueldo.joblib') 


# #### 8. Exportando las preguntas

# In[130]:


objects_sueldo = [freq_words, descripcion_m]


# In[131]:


objects_sueldo[0]


# In[132]:


objects_sueldo[1]


# In[133]:


joblib.dump(objects_sueldo, "objects_sueldo.joblib")


# In[ ]:





# In[134]:


descripcion_m

