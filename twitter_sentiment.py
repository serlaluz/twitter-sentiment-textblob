from twitter_scraper import get_tweets
import streamlit as st
import pandas as pd
from ast import literal_eval
from tqdm.notebook import tqdm
from time import sleep
import re
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import base64

plt.style.use('fivethirtyeight')

# Required Universal Functions



# Function to clean the data

def cleanTxt(text):
  text = re.sub(r'@[A-Za-z0-9]+', '', text) #remove @, r tells python its a raw string. Patterns we find. /
  text = re.sub(r'#', '', text) #remove hashtag
  text = re.sub(r'-', '', text) #removes hypens
  text = re.sub(r'RT[\s]+', '', text) #removing RT
  text = re.sub(r'https?:\/\/\S+', '', text) #remove the hyperlink
  return text

#These are functions for machine learning

#function to get subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

#create function to get polarity (how positive or negative text is)
def getPolarity(text):
  return TextBlob(text).sentiment.polarity 

#Global Text Variable
intro_description = '''This is a very basic app that will let you to search for information on twitter and then allows the ML to analyze the sentiment of the information from twitter. You can view results in different types of graphs'''
disclaimer = "### This app example is a demonstration of how to use textblob's sentiment analysis"
sources = '''* https://www.youtube.com/watch?v=ujId4ipkBio, Computer Science, 
* https://pypi.org/project/twitter-scraper/, Scrape Twitter'''

def main():
  '''This is the main script to run the app
  '''
  st.title("Twitter Sentiment Using Textblob(ML)")
  st.image("twitter.png", use_column_width = True)
  st.info(intro_description)
  st.write(disclaimer)

  #'''This section will allow the user to scrape twitter for content'''
  search_name = st.text_input("Search on twitter by @ = ID, # = hastag. e.g. @twitter, or #twitter. No spacing please", "@twitter")

  #Drop down menu of what is selectable
  #User will choose how many content they would want to scrape. 
  st.warning("Please note that the more pages you select, the longer the processing time.")
  pages = st.slider('Select how many pages you want to scrape', min_value = 1, max_value = 1000)

  #This will scrape twitter
 
  tweets = get_tweets(search_name, pages)
  tweets_df = pd.DataFrame()

  for tweet in tweets:
    print('Keys:', list(tweet.keys()), '\n')
    break

  for tweet in tweets:
    s = pd.DataFrame({'text' : [tweet['text']]})
    d = pd.DataFrame({'text' : [tweet['text']], 'date' : [tweet['time']]})
    tweets_print = tweets_df.append(d)
    tweets_df = tweets_df.append(s, ignore_index = True)

  #User will press button to see the scraped data
  if st.button("See Twitter Data"):
    st.write(tweets_print)

  #Call the clean function
  tweets_df['text'] = tweets_df['text'].apply(cleanTxt)

  # for tweet in tweets:
  # d = pd.DataFrame({'text' : [tweet['text']]})
  # tweets_df = tweets_df.append(s, ignore_index = True)

  
  csv_exp = tweets_print.to_csv(index=False)
  b64 = base64.b64encode(csv_exp.encode()).decode()  
  href = f'<a href="data:file/csv;base64,{b64}">Download Twitter Info</a> right-click and save link as **[file name].csv**'
  st.markdown(href, unsafe_allow_html=True)

  '''This section will allow the user to use the scraped twitter data and do the sentiment analysis.
  '''

  #User will press button to do text sentiment Analysis. 
  st.write("## Sentiment Analysis")
  #Ceates two new columns in the dataframe
  tweets_df['Subjectivity'] = tweets_df['text'].apply(getSubjectivity)
  tweets_df['Polarity'] = tweets_df['text'].apply(getPolarity)

  #This is function that calculates and classifies the numerical score into ['Negative', 'Neutral', and 'Positive']
  def getAnalysis(score):
    if score < 0:
      return 'Negative'
    elif score == 0:
      return 'Neutral'
    else:
      return 'Positive'

  tweets_df['Analysis'] = tweets_df['Polarity'].apply(getAnalysis)

  st.write(tweets_df)

  csv_exp1 = tweets_df.to_csv(index=False)
  b64 = base64.b64encode(csv_exp1.encode()).decode()  
  href = f'<a href="data:file/csv;base64,{b64}">Download Twitter Sentiment</a> right-click and save link as **[file name].csv**'
  st.markdown(href, unsafe_allow_html=True)

  user_choice = ['Show Bar Chart', 'Show Word Cloud', 'Show Percentages', 'Show Polarity Chart', 'Show Positive Sentiment', 'Show Negative Sentiment']
  choice = st.selectbox("Select Features", user_choice)  

  #User will press button to see the the word cloud
  if choice == 'Show Word Cloud':
    #word cloud shows how frequent words appear. The lager the text, the more frequent that word appeared in the tweet data set
    allWords = ' '.join( [twts for twts in tweets_df['text']] )
    wordCloud = WordCloud(width = 500, height = 300, random_state = 22, max_font_size = 120).generate(allWords)

    plt.imshow(wordCloud, interpolation = "bilinear")
    plt.axis('off')

    st.pyplot()

  #User will press button to see the positive sentiment
  if choice == 'Show Positive Sentiment':
    # display out all positive tweets
    j = 1
    sortedDF = tweets_df.sort_values(by=['Polarity'])
    for i in range(0, sortedDF.shape[0]):
      if (sortedDF['Analysis'][i] == 'Positive'):
        pos = str(j) + ') ' + sortedDF['text'][i]
        st.write(pos)
        j = j+1

  #User will press button to see the negative sentiment
  if choice == 'Show Negative Sentiment':
    # display out all negative tweets
    j = 1
    sortedDF = tweets_df.sort_values(by=['Polarity'], ascending ='False')
    for i in range(0, sortedDF.shape[0]):
      if (sortedDF['Analysis'][i] == 'Negative'):
        neg = str(j) + ') ' + sortedDF['text'][i]
        st.write(neg)
        j = j+1

  #user will press buton to see the polarity plot
  if choice == 'Show Polarity Chart':
    #plot the polarity chart 
    plt.figure(figsize=(8,6))
    for i in range(0, tweets_df.shape[0]):
      plt.scatter(tweets_df['Polarity'][i], tweets_df['Subjectivity'][i], color = "Blue")

    plt.title("Sentiment Analysis")
    plt.xlabel("Polarity")
    plt.ylabel("Subjectivity")
    st.pyplot()

  if choice == 'Show Percentages':
    #Gets the percentage of positive tweets
    ptweets = tweets_df[tweets_df.Analysis == 'Positive']
    ptweets = ptweets['text']

    percentage = round((ptweets.shape[0] / tweets_df.shape[0]) *100, 1)
    positives = "Positive Percentage : " + str(percentage) + "%"

    #Gets the percentage of negative tweets
    ntweets = tweets_df[tweets_df.Analysis == 'Negative']
    ntweets = ntweets['text']

    percentage_neg = round((ntweets.shape[0] / tweets_df.shape[0]) *100, 1)
    negatives = "Negative Percentage : " + str(percentage_neg) + "%"

    st.success(positives)
    st.error(negatives)

  #user will press button to see the bar chart (positive, negative, neutral)
  if choice == 'Show Bar Chart':
    sns.set(font_scale=1.4)
    tweets_df['Analysis'].value_counts()
    #plot the bar chart
    plt.title('Sentiment Analysis', y=1.02)
    plt.xlabel('Sentiment', labelpad=14)
    plt.ylabel('Counts', labelpad=14)
    tweets_df['Analysis'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0);
    st.pyplot()

  st.info("References and Sources")
  st.write(sources)

if __name__ == '__main__':
  main()
