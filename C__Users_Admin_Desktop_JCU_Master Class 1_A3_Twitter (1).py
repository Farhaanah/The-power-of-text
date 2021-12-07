#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q snscrape')


# In[2]:


import snscrape.modules.twitter as sntwitter
import pandas as pd


# In[15]:


cv_tweets = []

# Scraping data and append tweets to list
for i, tweet in enumerate(sntwitter.TwitterSearchScraper('coronavirus since:2021-11-26 until:2021-12-06').get_items()):
    cv_tweets.append([tweet.date, tweet.id, tweet.content])
    if i > 500:
        break
    #ref https://helloml.org/performing-sentiment-analysis-on-tweets-using-python/


# In[16]:


# Creating the dataframe, Export .csv file
cv_df = pd.DataFrame(cv_tweets, columns=['Datetime', 'Tweet Id', 'Text'])
cv_df.to_csv('coronavirus.csv')


# In[ ]:


#data preprocessing and cleaning 


# In[17]:


get_ipython().system('pip install tweet-preprocessor')


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessor as p


# In[190]:


df1 = pd.read_csv('coronavirus.csv',lineterminator='\n')
df1


# In[191]:


#drop columns "Tweet Id" and "Unnamed: 0"
df2 = df1.drop(["Tweet Id","Unnamed: 0"], axis=1)


# In[192]:


#using function from https://helloml.org/performing-sentiment-analysis-on-tweets-using-python/

def preprocess_tweet(text):
    text = p.clean(text)
    return text

#In the case of tweets data, it contains some twitter features such as RT tags (Retweet tags), user tags (@), hashtags (#), etc. Tweets may also contain links to external websites (URLs). These features do not affect the sentiment of the tweets and are unnecessary as well as redundant for our analysis. Tweets also contain some language-based features since on social media most of us make an informal (usage of multiple punctuations, emojis, words, unordered casings, etc.) and not a logical/structured use of language.


# In[193]:


#run df2 into cleaning function
df2['Clean Text'] = df2['Text\r'].apply(preprocess_tweet)
df2['Clean Text']= df2['Clean Text'].str.replace('[^\w\s]','')


# In[194]:


# Covert Datetime column type 
df2["Datetime"] = pd.to_datetime(df2["Datetime"]) #https://github.com/Bhasfe/distance_learning/blob/master/Distance_Learning.ipynb


# In[195]:


#remove duplicate rows
df2.drop_duplicates(inplace=True) #https://github.com/Bhasfe/distance_learning/blob/master/Distance_Learning.ipynb


# In[196]:


# Print the info again
print(df2.info()) #there were no duplicate rows 


# In[197]:


#remove stop words using natural language toolkit e.g. “I”, “me”, “my”, “myself”, “we”, “our”, “ours”, etc. Code adapted from https://helloml.org/performing-sentiment-analysis-on-tweets-using-python/
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')


# In[199]:


#apply stopwords function for english to this dataframe of tweets
df2['Clean Text'] = df2['Clean Text'] .apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#Code adapted from https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe


# In[200]:


#print df2 to ensure cleaning of text has worked
df2


# In[202]:


import re
def process_tweets(tweet):
    
    # Remove links
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    
    # Remove mentions and hashtag
    tweet = re.sub(r'\@\w+|\#','', tweet)
    
    # Tokenize the words
    tokenized = word_tokenize(tweet)

    # Remove the stop words
    tokenized = [token for token in tokenized if token not in stopwords.words("english")] 

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokenized = [lemmatizer.lemmatize(token, pos='a') for token in tokenized]

    # Remove non-alphabetic characters and keep the words contains three or more letters
    tokenized = [token for token in tokenized if token.isalpha() and len(token)>2]
    
    return tokenized
    
# Call the function and store the result into a new column
df2['Clean Text'] = df2['Clean Text'].str.lower().apply(process_tweets)

# Print the first fifteen rows of Processed
display(df2['Clean Text'].head(15))


# In[ ]:


#Performing Sentiment Analysis with VADER 


# In[30]:


nltk.download('vader_lexicon') # Download the VADER lexicon
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


# In[32]:


#Code apapted from https://helloml.org/performing-sentiment-analysis-on-tweets-using-python/

# Initialize sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Obtaining NLTK scores
df2['VScore'] = df2['Clean Text'].apply(lambda x: sia.polarity_scores(x))

# Obtaining NLTK compound score
df2['VComp'] = df2['VScore'].apply(lambda score_dict: score_dict['compound'])

# Set threshold to define neutral sentiment
neutral_thresh = 0.05

# Categorize scores into the sentiments of positive, neutral or negative
df2['Sentiment'] = df2['VComp'].apply(lambda c: 'Positive' if c >= neutral_thresh else ('Negative' if c <= -(neutral_thresh) else 'Neutral'))


# In[ ]:


#Generally, VADER is better for predicting sentiments for social network data, but as we need to establish the trends which data shows, we will use both the models and then compare the results. TextBlob gives us a polarity score, which helps us to classify sentiments. Similarly, VADER also gives a compound score, which is used to set classification labels.

#We set VADER compound score’s neutral threshold to be 0.05. Hence, we will use the following rule to classify sentiments:

#If -0.05 <= score <= 0.05: Tweet is of neutral sentiment.
#If -0.05 > score: Tweet is of negative sentiment.
#If 0.05 < score: Tweet is of positive sentiment.

#https://helloml.org/performing-sentiment-analysis-on-tweets-using-python/


# In[33]:


#examining polarity 
#Now, that we have assigned Sentiments to our tweets, we will plot a histogram/frequency plot of polarity scores to better understand the distribution of tweets among sentiment categories.
fig = plt.figure(figsize=(10, 6))
df2['VComp'].hist()
plt.xlabel('Polarity Score', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


# In[34]:


#look at the percentage of tweets across all sentiment categories.
def get_value_counts(col_name):
    count = pd.DataFrame(df2[col_name].value_counts())
    percentage = pd.DataFrame(df2[col_name].value_counts(normalize=True).mul(100))
    value_counts_df = pd.concat([count, percentage], axis = 1)
    value_counts_df = value_counts_df.reset_index()
    value_counts_df.columns = ['sentiment', 'counts', 'percentage']
    value_counts_df.sort_values('sentiment', inplace = True)
    value_counts_df['percentage'] = value_counts_df['percentage'].apply(lambda x: round(x,2))
    value_counts_df = value_counts_df.reset_index(drop = True)
    return value_counts_df
vader_sentiment_df = get_value_counts('Sentiment')
vader_sentiment_df


# In[37]:


#plot percentages of sentiment in frequency plot 
ax = sns.barplot(x="sentiment", y="percentage", data=vader_sentiment_df)
ax.set_title('Vader results')

for index, row in vader_sentiment_df.iterrows():
    ax.text(row.name,row.percentage, round(row.percentage,1), color='black', ha="center")


# In[ ]:


#though most of posts are neutral, this may also be due to some posts not being in english and therefore not being able to be classified into a negative or positive group.


# In[39]:


# creating a dict file to assign a number to each 
sentiment = {'Positive': 1,'Neutral': 3, 'Negative': 0}
#Code adapted from https://www.geeksforgeeks.org/replacing-strings-with-numbers-in-python-for-data-analysis/


# In[40]:


# Looping through data frame to change all sentiment names to a Sentiment Number based on dictionary defined above
df2.Sentiment = [sentiment[item] for item in df2.Sentiment]
print(df2) 
#Code adapted from https://www.geeksforgeeks.org/replacing-strings-with-numbers-in-python-for-data-analysis/


# In[41]:


#Drop all Neutral values in Sentiment column 
df3 = df2[df2.Sentiment != 3]


# In[46]:


#select only Clean Text and Sentiment Columns in a data frame
d4=df3[['Clean Text','Sentiment']]


# In[49]:


#seperating feature and target variable 
X=d4['Clean Text']
y=d4['Sentiment']


# In[59]:


# Separating the 95% data for training data and 5% for testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)
#Code adapted from https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/


# In[60]:


#transforming data based on TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))
#Code adapted from https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/


# In[61]:


#apply TfidfVectorizer to x_train and x_test
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
#Code adapted from https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/


# In[62]:


#Evaluate Model 
def model_Evaluate(model):
# Predict values for Test dataset
    y_pred = model.predict(X_test)
# Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
# Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


# In[63]:


#using Bernoulli Naive Bayes, a simple model classifier 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB


# In[65]:


BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)


# In[142]:


get_ipython().system('pip install wordcloud')


# In[143]:


from wordcloud import WordCloud


# In[148]:


df2


# In[149]:


data_neg = df2['Clean Text'][:800000]
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(wc)
#Code adapted from https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/


# In[155]:


import matplotlib as mpl
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
mpl.rcParams['figure.figsize']=(12.0,12.0)  
mpl.rcParams['font.size']=12            
mpl.rcParams['savefig.dpi']=100             
mpl.rcParams['figure.subplot.bottom']=.1 
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=500,
                          max_font_size=40, 
                          random_state=100
                         ).generate(str(df2['Clean Text']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show();
#wordcloud to simply visualize the data and which words are shown in varying sizes depending on how often they appear in Trump tweets.
#Code adapted from https://medium.datadriveninvestor.com/trump-tweets-topic-modeling-using-latent-dirichlet-allocation-e4f93b90b6fe


# In[208]:


from nltk.probability import FreqDist

#iterate through each tweet, then each token in each tweet, and store in one list
flat_words = [item for sublist in df2['Clean Text'] for item in sublist]

word_freq = FreqDist(flat_words)

word_freq.most_common(30)
#Code apadated https://towardsdatascience.com/lda-topic-modeling-with-tweets-deff37c0e131


# In[209]:


#retrieve word and count from FreqDist tuples

most_common_count = [x[1] for x in word_freq.most_common(30)]
most_common_word = [x[0] for x in word_freq.most_common(30)]

#create dictionary mapping of word count
top_30_dictionary = dict(zip(most_common_word, most_common_count))
#Code apadated https://towardsdatascience.com/lda-topic-modeling-with-tweets-deff37c0e131


# In[210]:


from wordcloud import WordCloud

#Create Word Cloud of top 30 words
wordcloud = WordCloud(colormap = 'Accent', background_color = 'black').generate_from_frequencies(top_30_dictionary)

#plot with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('top_30_cloud.png')

plt.show()
#Code apadated https://towardsdatascience.com/lda-topic-modeling-with-tweets-deff37c0e131


# In[213]:


pip install gensim


# In[214]:


import gensim
from gensim.corpora import Dictionary

#create dictionary
text_dict = Dictionary(df2['Clean Text'])

#view integer mappings
text_dict.token2id
#Code apadated https://towardsdatascience.com/lda-topic-modeling-with-tweets-deff37c0e131


# In[215]:


tweets_bow = [text_dict.doc2bow(tweet) for tweet in df2['Clean Text']]
#Code apadated https://towardsdatascience.com/lda-topic-modeling-with-tweets-deff37c0e131


# In[216]:


tweets_bow
#Code apadated https://towardsdatascience.com/lda-topic-modeling-with-tweets-deff37c0e131


# In[217]:


from gensim.models.ldamodel import LdaModel

k = 5
tweets_lda = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)

tweets_lda.show_topics()
#Code apadated https://towardsdatascience.com/lda-topic-modeling-with-tweets-deff37c0e131


# In[221]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[228]:


from gensim.models.ldamodel import LdaModel

k = 6
tweets_lda2 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[229]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda2, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[230]:


from gensim.models.ldamodel import LdaModel

k = 7
tweets_lda3 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[231]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda3, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[232]:


from gensim.models.ldamodel import LdaModel

k = 8
tweets_lda4 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[233]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda4, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[234]:


from gensim.models.ldamodel import LdaModel

k = 9
tweets_lda5 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[235]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda5, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[236]:


from gensim.models.ldamodel import LdaModel

k = 10
tweets_lda6 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[237]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda6, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[238]:


from gensim.models.ldamodel import LdaModel

k = 11
tweets_lda7 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[239]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda7, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[240]:


from gensim.models.ldamodel import LdaModel

k = 12
tweets_lda8 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[241]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda8, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[242]:


from gensim.models.ldamodel import LdaModel

k = 13
tweets_lda9 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[243]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda9, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[244]:


from gensim.models.ldamodel import LdaModel

k = 4
tweets_lda10 = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


# In[245]:


#calculating coherence score
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=tweets_lda10, texts= df2['Clean Text'], dictionary=text_dict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Code adapted from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

