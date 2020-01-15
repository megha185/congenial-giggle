#!/usr/bin/env python
# coding: utf-8

# In[2]:


#First, we import some Python packages that will help us analyzing the data,especially pandas for data analysis and matplotlib for visualization
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns


import warnings
from collections import Counter
import datetime
import wordcloud
import json


# In[3]:


# Hiding warnings for cleaner display
warnings.filterwarnings('ignore')

# Configuring some options
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# If you want interactive plots, uncomment the next line
# %matplotlib notebook


# In[4]:


#Then we read the dataset file which is in csv format
df = pd.read_csv(r"C:\Users\Nawed\Desktop\INvideos.csv")


# In[5]:


df.head(10)


# In[6]:


PLOT_COLORS = ["#268bd2", "#0052CC", "#FF5722", "#b58900", "#003f5c"]
pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('font', family='Arial', weight='400', size=10)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)


# In[7]:


df.info()


# In[8]:


#he description column has some null values(561 rows). These are some of the rows whose description values are null. We can see that null values are denoted by NaN
#data cleaning
df[df["description"].apply(lambda x: pd.isna(x))].head(3)


# In[10]:


#So to do some sort of data cleaning, and to get rid of those null values, we put an empty string in place of each null value in the description column
df["description"] = df["description"].fillna(value="")


# In[11]:


#Let's see in which years the dataset was collected
cdf = df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts()             .to_frame().reset_index()             .rename(columns={"index": "year", "trending_date": "No_of_videos"})

fig, ax = plt.subplots()
_ = sns.barplot(x="year", y="No_of_videos", data=cdf, 
                palette=sns.color_palette(['#ff764a', '#ffa600'], n_colors=7), ax=ax)
_ = ax.set(xlabel="Year", ylabel="No. of videos")


# In[12]:


df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts(normalize=True)


# In[ ]:


#We can see that the dataset was collected in 2017 and 2018 with 76% of it in 2018 and 24% in 2017.


# In[13]:


#Now, let's see some statistical information about the numerical columns of our dataset
df.describe()


# In[ ]:


#We note from the table above that
#The average number of views of a trending video is 1,060,477. The median value for the number of views is 304,586
#The average number of likes of a trending video is 27,082, while the average number of dislikes is 1,665. 
#The average comment count is 2,677 while the median is 329.


# In[ ]:


# Now let's plot a histogram for the views column to take a look at its distribution: to see how many videos have between 10 million and 20 million views, how many videos have between 20 million and 30 million views, and so on.


# In[14]:


fig, ax = plt.subplots()
_ = sns.distplot(df["views"], kde=False, color=PLOT_COLORS[4], 
                 hist_kws={'alpha': 1}, bins=np.linspace(0, 2.3e8, 47), ax=ax)
_ = ax.set(xlabel="Views", ylabel="No. of videos", xticks=np.arange(0, 2.4e8, 1e7))
_ = ax.set_xlim(right=2.5e8)
_ = plt.xticks(rotation=90)


# In[ ]:


#We note that the vast majority of trending videos have 5 million views or less. We get the 5 million number by calculating

[0.1×(10)8]/2 = 5*(10)6
 
#Now let us plot the histogram just for videos with 25 million views or less to get a closer look at the distribution of the data


# In[15]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["views"] < 25e6]["views"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Views", ylabel="No. of videos")


# In[ ]:


#Now we see that the majority of trending videos have 1 million views or less. Let's see the exact percentage of videos less than 1 million views


# In[16]:


df[df['views'] < 1e6]['views'].count() / df['views'].count() * 100


# In[ ]:


#So, it is around 80%.


# In[17]:


plt.rc('figure.subplot', wspace=0.9)
fig, ax = plt.subplots()
_ = sns.distplot(df["likes"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, 
                 bins=np.linspace(0, 6e6, 61), ax=ax)
_ = ax.set(xlabel="Likes", ylabel="No. of videos")
_ = plt.xticks(rotation=90)


# In[ ]:


#We note that the vast majority of trending videos have between 0 and 100,000 likes. Let us plot the histogram just for videos with 1000,000 likes or less to get a closer look at the distribution of the data


# In[18]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["likes"] <= 1e5]["likes"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Likes", ylabel="No. of videos")


# In[ ]:


#Now we can see that the majority of trending videos have 20000 likes or less with a peak for videos with 5000 likes or less.


# In[46]:


df[df['likes'] < 4e4]['likes'].count() / df['likes'].count() * 100


# In[47]:


#Comment count histogram
fig, ax = plt.subplots()
_ = sns.distplot(df["comment_count"], kde=False, rug=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Comment Count", ylabel="No. of videos")


# In[48]:


#Let's get a closer look by eliminating entries with comment count larger than 200000 comment

fig, ax = plt.subplots()
_ = sns.distplot(df[df["comment_count"] < 200000]["comment_count"], kde=False, rug=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, 
                 bins=np.linspace(0, 2e5, 49), ax=ax)
_ = ax.set(xlabel="Comment Count", ylabel="No. of videos")


# In[ ]:


#We see that most trending videos have around

# 25000/6≈4166 comments
 
#since each division in the graph has six histogram bins.

#As with views and likes, let's see the exact percentage of videos with less than 4000 comments


# In[49]:


df[df['comment_count'] < 4000]['comment_count'].count() / df['comment_count'].count() * 100


# In[20]:


#Description on non-numerical columns
df.describe(include = ['O'])


# In[ ]:


#Now we want to see how many trending video titles contain at least a capitalized word (e.g. HOW). To do that, we will add a new variable (column) to the dataset whose value is True if the video title has at least a capitalized word in it, and False otherwise


# In[22]:


def contains_capitalized_word(s):
    for w in s.split():
        if w.isupper():
            return True
    return False


df["contains_capitalized"] = df["title"].apply(contains_capitalized_word)

value_counts = df["contains_capitalized"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'}, startangle=45)
_ = ax.axis('equal')
_ = ax.set_title('Title Contains Capitalized Word?')


# In[24]:


df["contains_capitalized"].value_counts(normalize=True)


# In[ ]:


#We can see that 40% of trending video titles contain at least a capitalized word. We will later use this added new column contains_capitalized in analyzing correlation between variables.


# In[25]:


#Video title lengths
#Let's add another column to our dataset to represent the length of each video title, then plot the histogram of title length to get an idea about the lengths of trnding video titles

df["title_length"] = df["title"].apply(lambda x: len(x))

fig, ax = plt.subplots()
_ = sns.distplot(df["title_length"], kde=False, rug=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Title Length", ylabel="No. of videos", xticks=range(0, 110, 10))


# In[26]:


fig, ax = plt.subplots()
_ = ax.scatter(x=df['views'], y=df['title_length'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Views", ylabel="Title Length")


# In[ ]:


#By looking at the scatter plot, we can say that there is no relationship between the title length and the number of views.


# In[ ]:


#Now let's see how the dataset variables are correlated with each other: for example, we would like to see how views and likes are correlated, meaning do views and likes increase and decrease together (positive correlation)? Does one of them increase when the other decrease and vice versa (negative correlation)? Or are they not correlated?

#Correlation is represented as a value between -1 and +1 where +1 denotes the highest positive correlation, -1 denotes the highest negative correlation, and 0 denotes that there is no correlation.

#Let's see the correlation table between our dataset variables (numerical and boolean variables only)


# In[27]:


df.corr()


# In[ ]:


#We see for example that views and likes are highly positively correlated with a correlation value of 0.85; we see also a high positive correlation (0.78) between likes and comment count, and between dislikes and comment count (0.71).

#There is some positive correlation between views and dislikes, between views and comment count, between likes and dislikes.

#Now let's visualize the correlation table above using a heatmap


# In[28]:


h_labels = [x.replace('_', ' ').title() for x in 
            list(df.select_dtypes(include=['number', 'bool']).columns.values)]

fig, ax = plt.subplots(figsize=(10,6))
_ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)


# In[ ]:


#The correlation map and correlation table above say that views and likes are highly positively correlated. Let's verify that by plotting a scatter plot between views and likes to visualize the relationship between these variables


# In[29]:


fig, ax = plt.subplots()
_ = plt.scatter(x=df['views'], y=df['likes'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Views", ylabel="Likes")


# In[ ]:


#We see that views and likes are truly positively correlated: as one increases, the other increases too—mostly.

#Another verification of the correlation matrix and map is the scatter plot we drew above between views and title length as it shows that there is no correlation between them.


# In[ ]:


#Most common words in video titles
#Let's see if there are some words that are used significantly in trending video titles. We will display the 25 most common words in all trending video titles


# In[30]:


title_words = list(df["title"].apply(lambda x: x.split()))
title_words = [x for y in title_words for x in y]
Counter(title_words).most_common(25)


# In[ ]:


#Let's draw a word cloud for the titles of our trending videos, which is a way to visualize most common words in the titles; the more common the word is, the bigger its font size is


# In[31]:


# wc = wordcloud.WordCloud(width=1200, height=600, collocations=False, stopwords=None, background_color="white", colormap="tab20b").generate_from_frequencies(dict(Counter(title_words).most_common(150)))

wc = wordcloud.WordCloud(width=1200, height=500, 
                         collocations=False, background_color="white", 
                         colormap="tab20b").generate(" ".join(title_words))
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation='bilinear')
_ = plt.axis("off")


# In[32]:


#Which channels have the largest number of trending videos?
cdf = df.groupby("channel_title").size().reset_index(name="video_count")     .sort_values("video_count", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8,8))
_ = sns.barplot(x="video_count", y="channel_title", data=cdf,
                palette=sns.cubehelix_palette(n_colors=20, reverse=True), ax=ax)
_ = ax.set(xlabel="No. of videos", ylabel="Channel")


# In[ ]:


#Which video category has the largest number of trending videos?


# In[33]:


with open(r"C:\Users\Nawed\Desktop\IN_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df['category_name'] = df['category_id'].map(cat_dict)


# In[34]:


cdf = df["category_name"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "category_name", "category_name": "No_of_videos"}, inplace=True)
fig, ax = plt.subplots()
_ = sns.barplot(x="category_name", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=16, reverse=True), ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
_ = ax.set(xlabel="Category", ylabel="No. of videos")


# In[ ]:


#We see that the Entertainment category contains the largest number of trending videos among other categories, followed by News & Politics , followed by Music and so on.


# In[35]:


#Trending videos and their publishing time
#An example value of the publish_time column in our dataset is 2017-11-13T17:13:01.000Z
#This means that the date of publishing the video is 2017-11-13 and the time is 17:13:01 in Coordinated Universal Time (UTC) time zone.

df["publishing_day"] = df["publish_time"].apply(
    lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d").date().strftime('%a'))
df["publishing_hour"] = df["publish_time"].apply(lambda x: x[11:13])
df.drop(labels='publish_time', axis=1, inplace=True)


# In[36]:


#Now we can see which days of the week had the largest numbers of trending videos

cdf = df["publishing_day"].value_counts()        .to_frame().reset_index().rename(columns={"index": "publishing_day", "publishing_day": "No_of_videos"})
fig, ax = plt.subplots()
_ = sns.barplot(x="publishing_day", y="No_of_videos", data=cdf, 
                palette=sns.color_palette(['#003f5c', '#374c80', '#7a5195', 
                                           '#bc5090', '#ef5675', '#ff764a', '#ffa600'], n_colors=7), ax=ax)
_ = ax.set(xlabel="Publishing Day", ylabel="No. of videos")


# In[ ]:


#We can see that the number of trending videos published on Sunday and Wednesday are noticeably less than the number of trending videos published on other days of the week.


# In[37]:


#Now let's use publishing_hour column to see which publishing hours had the largest number of trending videos

cdf = df["publishing_hour"].value_counts().to_frame().reset_index()        .rename(columns={"index": "publishing_hour", "publishing_hour": "No_of_videos"})
fig, ax = plt.subplots()
_ = sns.barplot(x="publishing_hour", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=24), ax=ax)
_ = ax.set(xlabel="Publishing Hour", ylabel="No. of videos")


# In[38]:


#How many trending videos have an error?

value_counts = df["video_error_or_removed"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
        colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Video Error or Removed?')


# In[39]:


df["video_error_or_removed"].value_counts()


# In[ ]:


#We can see that out of videos that appeared on trending list (37341 videos), there is a tiny portion (11 videos) with errors.


# In[40]:


#How many trending videos have their commets disabled?

value_counts = df["comments_disabled"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie(x=[value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Comments Disabled?')


# In[41]:


df["comments_disabled"].value_counts(normalize=True)


# In[ ]:


#We see that only 3% of trending videos prevented users from commenting.


# In[42]:


#How many trending videos have their ratings disabled?

value_counts = df["ratings_disabled"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
            colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Ratings Disabled?')


# In[43]:


df["ratings_disabled"].value_counts()


# In[ ]:


#We see that only 781 trending videos out of 36571 prevented users from commenting.


# In[44]:


#How many videos have both comments and ratings disabled?

len(df[(df["comments_disabled"] == True) & (df["ratings_disabled"] == True)].index)


# In[ ]:


#So there are just 360 trending videos that have both comments and ratings disabled

