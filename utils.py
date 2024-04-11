from nltk.corpus import stopwords
import numpy as np

#Converting everything into lowertext
def conv_lower(text):
  return text.lower()

#Removing special characters
def remove_special(text):
  x=''
  for i in text:
    if i.isalnum():
      x=x+i
    else:
      x=x+' '
  return x

def remove_stopwords(text):
  x=[]
  for i in text.split():
    if i not in stopwords.words('english'):
      x.append(i)
  y=x[:]
  x.clear()
  return y

#converting list back to strings
def join_back(list_input):
  return ' '.join(list_input)

def document_vector(doc,model):
    # remove out-of-vocabulary words
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0)