from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
import pandas as pd
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Embedding,MaxPooling1D,GRU,Bidirectional
from keras.layers import Dropout,Conv1D,SpatialDropout1D,GlobalMaxPooling1D
from keras.optimizers import Adam
desired_width=320

pd.set_option('display.width', desired_width)

pd.set_option('display.max_columns',10)
df = pd.read_csv('data.txt',
                   header = None,
                delimiter='\t')
print(df)
df.columns = ['texte']

vader_score = []
vader_class = []

for review in df['texte']:

    blob = TextBlob(review, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    compound_score = blob.sentiment
    compound_score = compound_score[0]
    vader_score.append(compound_score)

    if (compound_score >= -1) and (compound_score < -0.7):
        vader_class.append("tres tres mecontent")

    elif (compound_score >= -0.7) and (compound_score < -0.35):
        vader_class.append("tres mecontent")

    elif (compound_score >= -0.35) and (compound_score < 0):
        vader_class.append("mecontent")

    elif (compound_score == 0):
        vader_class.append("neutre")

    elif (compound_score > 0) and (compound_score < 0.35):
        vader_class.append("content")

    elif (compound_score >= 0.35) and (compound_score < 0.7):
        vader_class.append("tres content")
    elif (compound_score >= 0.7) and (compound_score <= 1):
        vader_class.append("tres tres content")

df['polarity_score'] = vader_score
df['sentiment'] = vader_class

# print(df.head(20))
df["ettiquete"] = None
df.loc[df['sentiment'] == 'tres tres mecontent', 'ettiquete'] = 0
df.loc[df['sentiment'] == 'tres mecontent', 'ettiquete'] = 1
df.loc[df['sentiment'] == 'mecontent', 'ettiquete'] = 2
df.loc[df['sentiment'] == 'neutre', 'ettiquete'] = 3
df.loc[df['sentiment'] == 'content', 'ettiquete'] = 4
df.loc[df['sentiment'] == 'tres content', 'ettiquete'] = 5
df.loc[df['sentiment'] == 'tres tres content', 'ettiquete'] = 6
data = df.to_csv('new_data_french.txt', header=True, index=False, sep='\t')
#________________________________________________________________________________________________ PART 2___________________________________________________________________________________________________________#

data = pd.read_csv('new_data_french.txt',delimiter='\t')
#print(data)
import re
from nltk.corpus import stopwords
data = data.reset_index(drop=True)
STOPWORDS = set(stopwords.words('french'))
def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(r"\d", "", text) #remove number
    text = re.sub(r"\s+", " ", text, flags=re.I) # remove space
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text
data['texte'] = data['texte'].apply(clean_text)

print(data.head(6))

X=data.loc[:, 'texte'].values

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 360
# This is fixed.
EMBEDDING_DIM = 100
# --------pas de sequence----------------#

from tensorflow.python.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='0123456789!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
print(tokenizer)
tokenizer.fit_on_texts(data['texte'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
max_lenght=max([len(s.split()) for s in X]) # la longueur maximale de phrases d'entrée
print("nombre max de sequence d'entrer= ", max_lenght)
vocab_size=len(tokenizer.word_index)+1
print("vocab size = ", vocab_size) # nombre de vocabulaire
print("dictionnaire_index = ", tokenizer.word_index)
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(data['texte'].values)
X = pad_sequences(X, maxlen= max_lenght)
print(X.shape)

y=data.loc[:, 'ettiquete'].values
print((Counter(y)))
y = to_categorical(y)

# deviser data
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


#--------------------------------------------------------------------------------------------------#

embedding_vector_length = 64
opt =Adam(lr=0.0009)
import tensorflow as tf
optimizer = tf.keras.optimizers.RMSprop (0.0099)
from keras import regularizers
opt =Adam(lr=0.0001)
embedding_vector_length = 64
epochs = 28
batch_size = 64
filter_length = 16
from keras.regularizers import l2
from sklearn.utils import class_weight

model = Sequential()
model.add(Embedding(input_dim= vocab_size,  output_dim = embedding_vector_length, input_length=max_lenght))
model.add(SpatialDropout1D(0.1))
model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.1))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(7, activation='softmax'))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stops = EarlyStopping(patience=3, monitor='val_loss',min_delta=0.001)
checkpointer = ModelCheckpoint(filepath='weights4.best.eda.hdf5', verbose=1, save_best_only=True)
history =model.fit(X_train, Y_train,validation_data=(X_test,Y_test) ,batch_size = 64, epochs = 100,callbacks=[checkpointer], verbose=1)
from sklearn.utils import class_weight

# SAVE BON MODEL
model.save('nnfl4.eda.hdf5')
# EVALUATION
batch_size = 64
score,acc = model.evaluate(X_test, Y_test, batch_size = batch_size)
print("Score: %.2f" % (score))
print("Validation Accuracy: %.2f" % (acc))


# la valeur de perte et les valeurs de métrique
scores1 = model.evaluate(X_train, Y_train)
scores2 = model.evaluate(X_test, Y_test)
print("Accuracy_train: %.2f%%" % (scores1[1]*100)) # precision du formation
 #performances de modèle sur des données de test
print("Accuracy_test: %.2f%%" % (scores2[1]*100)) # precision du validation



from sklearn.metrics import classification_report
preds = model.predict(X_test)
preds_train=model.predict(X_train)
print("classification_report_Test \n", classification_report(np.argmax(Y_test,axis=1),np.argmax(preds,axis=1)))
print("_________________________________________________________________")
print("classification_report_Train \n", classification_report(np.argmax(Y_train,axis=1),np.argmax(preds_train,axis=1)))

# make class predictions with the model
predictions = model.predict_classes(X_train)
# summarize the first 5 cases
for i in range(10):
	print(  (X_train[i].tolist(), " classe predecte => " , predictions[i]))
#----------------------------------test--------------------------------------------------------#



