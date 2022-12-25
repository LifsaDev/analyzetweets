import streamlit as st 
import numpy as np 
import tweepy
import json
 
import pandas as pd 
import matplotlib.pyplot as plt 
import re 
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

 
 

def get_tweets(topic):
   consumer_key = "Tba4dz6iAhJ8aIlruIVyOgv1H"
   consumer_secret = "hrMWxGtZPKjDPhWz6iPVRIXnrJAFofRqW8cCh3lQG9TKXSNUEs"
   access_token = "1250167730981998592-RVTbFdCcrbvxl4VWOPh1MMrur0Co0w"
   access_token_secret = "n5K7kaJDMtMChQBqoMtPX957NJNdOyj1nY1O9Kapv5vZk"

   # Créer une connexion à l'API de Twitter en utilisant les clés d'accès et les jetons d'accès
   auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
   api = tweepy.API(auth)

   # Définir le terme de recherche à utiliser pour récupérer les tweets
   query = topic

   # Récupérer les tweets correspondant au terme de recherche
   tweets = api.search_tweets(q=query, lang='en', count=100)
 
   # Parcourir chaque tweet et ajouter les données au flux de données
   df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
   i=0
   try:
    for tweet in tweets:
        df.loc[i,"Date"] = tweet.created_at
        df.loc[i,"User"] = tweet.user.name
        df.loc[i,"IsVerified"] = tweet.user.verified
        df.loc[i,"Tweet"] = tweet.text
        df.loc[i,"Likes"] = tweet.favorite_count
        df.loc[i,"RT"] = tweet.retweet_count
        df.loc[i,"User_location"] = tweet.user.location
        i=i+1 
        if i>100:
            break
        else:
            pass 
   except Exception as e:
    pass 
   return df 

# clean tweets
def clean_tweet(tweet):
    txt_cl=' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])',
                           ' ',str(tweet).lower()).split())
    return txt_cl 
#prepCloud words 
def prepCloud(Topic_text,Topic):
    Topic = str(Topic).lower()
    Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
    Topic = re.split("\s+",str(Topic))
    stopwords = set(STOPWORDS)
    stopwords.update(Topic) # Add our topic in Stopwords, so it doesnt appear in wordClous
    ###
    text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
    return text_new
#get subjectivity
def get_subjectivity(tweet):
    analysis = TextBlob(tweet)
    subj=analysis.sentiment.subjectivity
    return subj
#get polarity
def get_polarity(tweet):
    analysis = TextBlob(tweet)
    pol=analysis.sentiment.polarity
    return pol 
#get sentiment
def get_sentiment(tweet):
    pol=get_polarity(tweet)
    if  pol > 0:
        return 'Positive'
    elif pol == 0:
        return 'Neutral'
    else:
        return 'Negative' 
#plot total wordcloud
def total_wordsCloud(df,max_words,Topic):
    text = " ".join(review for review in df.clean_tweet)

    # Create stopword list:
    stopwords = set(STOPWORDS)
    text_newALL = prepCloud(text,Topic)
    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords,max_words=max_words,max_font_size=70).generate(text_newALL)

    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("The most frequently used words when searching for {}".format(Topic),)
    plt.axis("off")
    plt.show()
#plot positive words clouds
def positive_wCloud(df,max_words,Topic):
    text_positive = " ".join(review for review in df[df["Sentiment"]=="Positive"].clean_tweet)
     
    # Create stopword list:
    stopwords = set(STOPWORDS)
    
    text_new_positive = prepCloud(text_positive,Topic)
 
    wordcloud = WordCloud(stopwords=stopwords,max_words=max_words,max_font_size=70).generate(text_new_positive)
 

    fig1=plt.figure(figsize=(8,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("The most positive words used when searching for {}".format(Topic))
    plt.axis("off")
    plt.show()
    st.pyplot(fig1)

# plot negative wCloud
def negative_wCloud(df,max_words,Topic):
    text_negative = " ".join(review for review in df[df["Sentiment"]=="Negative"].clean_tweet)
    stopwords = set(STOPWORDS) 
    text_new_negative = prepCloud(text_negative,Topic)

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords,max_words=max_words,max_font_size=70).generate(text_new_negative)
    fig2=plt.figure(figsize=(8,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("The most frequently used words when searching for {}".format(Topic))
    plt.axis("off")
    plt.show()
    st.pyplot(fig2)
#plot pie 
def plot_pie(df):
    a=len(df[df["Sentiment"]=="Positive"])
    b=len(df[df["Sentiment"]=="Negative"])
    c=len(df[df["Sentiment"]=="Neutral"])
    d=np.array([a,b,c])
    explode = (0.1, 0.0, 0.1)
    fig = plt.figure(figsize=(8,5))
    plt.pie(d,shadow=True,explode=explode,labels=["Positive","Negative","Neutral"],autopct='%1.2f%%');
    st.pyplot(fig)

def app():
    # Set page title
    st.set_page_config(
    page_title="analysetweet",
    page_icon="🧊")
    st.title('🌐Analyze Tweets From Twitter🌐')
    # Ajoutez votre description ici
    st.sidebar.title("🧊🧊__Team__🧊🧊")
    description ="🧊SAOUADOGO SALIFOU________________ \
                  🧊OUEDRAOGO OUSSENI________________\
                  🧊AYITE KOMLAN PAUL_________________\
                  🧊DIASSANA S.FATOUMATA______________ "

    st.sidebar.markdown(description)
    st.sidebar.title("Tools")
    topic=st.sidebar.text_input("Give the Topic")
    display_btn=st.sidebar.button("Display Tweets")
    analyze_btn=st.sidebar.button("Analyze Tweets")
     
    st.subheader("Analyze the tweets of your favourite Topic")
    st.subheader("This tool performs the following tasks :")

    st.write("🥇️ Get 100  most recent tweets")
    st.write("🥇️ Generates a Word Cloud")
    st.write("🥇️ Performs Sentiment Analysis")
     
    if display_btn:
        if topic!="":
            df=get_tweets(topic)
            if not df.empty:
                st.title("100 recent tweets for "+str(topic))
                st.write(df)
                df.to_csv("df.csv",index=False)
                pd.DataFrame({"topic":[topic]},index=[0]).to_csv("topic.csv")
    if analyze_btn:
        df=pd.read_csv("df.csv")
        print(df) 
        tp=str(pd.read_csv("topic.csv")["topic"].values[0])
        if topic!="":
            if tp==topic:
                st.title("100 recent tweets for "+str(topic))
                st.write(df)
                df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))
                df["Subjectivity"] = df["Tweet"].apply(lambda x : get_subjectivity(x))
                df["Polarity"] = df["Tweet"].apply(lambda x : get_polarity(x))
                df["Sentiment"] = df["Tweet"].apply(lambda x : get_sentiment(x))
                st.title("========After Analyze==========")
                st.write(df) 
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.header("Positive")
                    t1=len(df[df["Sentiment"]=="Positive"])
                    st.markdown(f"<div style='background-color: green;'><h1><center>{t1}</center></h1></div>", unsafe_allow_html=True)
                with col2:
                    st.header("Neutral")
                    t2=len(df[df["Sentiment"]=="Neutral"])
                    st.markdown(f"<div style='background-color:yellow;'><h1><center>{t2}</center></h1></div>", unsafe_allow_html=True)
                with col3:
                    st.header("Negative")
                    t3=len(df[df["Sentiment"]=="Negative"])
                    st.markdown(f"<div style='background-color: red;'><h1><center>{t3}</center></h1></div>", unsafe_allow_html=True)
                st.title("========Plot Pie==========")
                plot_pie(df)
                st.title("==Positive WordsCloud==")
                positive_wCloud(df,100,topic)
                st.title("==Negative WordsCloud==")
                negative_wCloud(df,100,topic)




if __name__=="__main__":
    app()