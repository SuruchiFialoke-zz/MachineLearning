
# Natural Language Processing on Hacker News Data



Natural language processing (NLP) is the study of enabling computers to understand human languages. This field may involve teaching computers to automatically score essays, infer grammatical rules, or determine the emotions associated with text.  In this project we will employ NLP on Hacker News data. Hacker News is a community where users can submit articles, and other users can upvote those articles. The articles with the most upvotes make it to the front page, where they're more visible to the community. We'll be predicting the number of upvotes the articles received, based on their headlines. Because upvotes are an indicator of popularity, we'll discover which types of articles tend to be the most popular.


Resources:

[View the project and data-files on DataQuest](https://www.dataquest.io/m/67/introduction-to-natural-language-processing/)
[Hacker News](https://news.ycombinator.com/)
[Scikit-learn Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)

Organization:


[Loading the Data](#loading-the-data)

[Bag of Words Model to Tokenize the Data](#bag-of-words-model-to-tokenize-the-data)

[Linear Regression](#linear-regression)


## Loading the Data

This set consists of submissions users made to Hacker News from 2006 to 2015. Developer Arnaud Drizard used the Hacker News API to scrape the data ([hosted on his github repository](https://github.com/arnauddri/hn)). DataQuest has randomly sampled 3000 rows from the [data, and removed all but 4 of the columns](https://www.dataquest.io/m/67/introduction-to-natural-language-processing/2/overview-of-the-data). 

Our data only has four columns:

- submission_time - When the article was submitted
- upvotes - The number of upvotes the article received
- url - The base URL of the article
- headline - The article's headline


```python
import pandas as pd

submissions = pd.read_csv("sel_hn_stories.csv")
submissions.columns = ["submission_time", "upvotes", "url", "headline"]
submissions = submissions.dropna()
```


```python
submissions.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>submission_time</th>
      <th>upvotes</th>
      <th>url</th>
      <th>headline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-02-17T16:57:59Z</td>
      <td>1</td>
      <td>blog.jonasbandi.net</td>
      <td>Software: Sadly we did adopt from the construc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-02-04T02:36:30Z</td>
      <td>1</td>
      <td>blogs.wsj.com</td>
      <td>Google’s Stock Split Means More Control for L...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-10-26T07:11:29Z</td>
      <td>1</td>
      <td>threatpost.com</td>
      <td>SSL DOS attack tool released exploiting negoti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-04-03T15:43:44Z</td>
      <td>67</td>
      <td>algorithm.com.au</td>
      <td>Immutability and Blocks Lambdas and Closures</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-13T16:49:20Z</td>
      <td>1</td>
      <td>winmacsofts.com</td>
      <td>Comment optimiser la vitesse de Wordpress?</td>
    </tr>
  </tbody>
</table>
</div>



## Bag of Words Model to Tokenize the Data

Our final goal is to train a linear regression algorithm that predicts the number of upvotes a headline would receive. To do this, we'll need to convert each headline to a numerical representation.

While there are several ways to accomplish this, we'll use a [bag of words model](https://en.wikipedia.org/wiki/Bag-of-words_model). A bag of words model represents each piece of text as a numerical vector as shown below.

![](https://image.slidesharecdn.com/wordembedings-whythehype-151203065649-lva1-app6891/95/word-embeddings-why-the-hype-4-638.jpg?cb=1449126428)


Step 1: Ttokenization or breaking a sentence up into disconnected words.

Step 2: Preprocessing Tokens To Increase Accuracy: Lowercasing and removing punctuation 

Step 3: Retrieve all of the unique words from all of the headlines

Step 4: Counting Token Occurrences


```python
# Step 1: Ttokenization
"""Split each headline into individual words on the space character(" "), 
and append the resulting list to tokenized_headlines."""

tokenized_headlines = []
for item in submissions["headline"]:
    tokenized_headlines.append(item.split(" "))
    
print(tokenized_headlines[:2])    
```

    [['Software:', 'Sadly', 'we', 'did', 'adopt', 'from', 'the', 'construction', 'analogy'], ['', 'Google’s', 'Stock', 'Split', 'Means', 'More', 'Control', 'for', 'Larry', 'and', 'Sergey', '']]



```python
# Step 2: Lowercasing and removing punctuation
"""For each list of tokens: Convert each individual token to lowercase
and remove all of the items from the punctuation list"""

punctuations_list = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []
for item in tokenized_headlines:
    tokens = []
    for token in item:
        token = token.lower()
        for punc in punctuations_list:
            token = token.replace(punc, "")
        tokens.append(token)
    clean_tokenized.append(tokens)

print(clean_tokenized[:2])     
```

    [['software', 'sadly', 'we', 'did', 'adopt', 'from', 'the', 'construction', 'analogy'], ['', 'googles', 'stock', 'split', 'means', 'more', 'control', 'for', 'larry', 'and', 'sergey', '']]



```python
# Step 3: Retrieve all of the unique words from all of the headlines
# unique_tokens contains any tokens that occur more than once across all of the headlines.

import numpy as np
unique_tokens = []
single_tokens = []
for tokens in clean_tokenized:
    for token in tokens:
        if token not in single_tokens:
            single_tokens.append(token)
        elif token in single_tokens and token not in unique_tokens:
            unique_tokens.append(token)

counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)
```


```python
# Step 4: Counting Token Occurrences
for i, item in enumerate(clean_tokenized):
    for token in item:
        if token in unique_tokens:
            counts.iloc[i][token] += 1
```


```python
counts.shape
counts.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>and</th>
      <th>for</th>
      <th>as</th>
      <th>you</th>
      <th>is</th>
      <th>the</th>
      <th>split</th>
      <th>good</th>
      <th>how</th>
      <th>...</th>
      <th>frameworks</th>
      <th>animated</th>
      <th>walks</th>
      <th>auctions</th>
      <th>clouds</th>
      <th>hammer</th>
      <th>autonomous</th>
      <th>vehicle</th>
      <th>crowdsourcing</th>
      <th>disaster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2309 columns</p>
</div>



### Removing Columns To Increase Accuracy

We have 2309 columns in our matrix, and too many columns will cause the model to fit to noise instead of the signal in the data. There are two kinds of features that will reduce prediction accuracy. Features that occur only a few times will cause overfitting, because the model doesn't have enough information to accurately decide whether they're important. These features will probably correlate differently with upvotes in the test set and the training set.

Features that occur too many times can also cause issues. These are words like and and to, which occur in nearly every headline. These words don't add any information, because they don't necessarily correlate with upvotes. These types of words are sometimes called stopwords.

To reduce the number of features and enable the linear regression model to make better predictions, we'll remove any words that occur fewer than 5 times or more than 100 times.


```python
word_counts = counts.sum(axis=0)
counts = counts.loc[:,(word_counts >= 5) & (word_counts <= 100)]
```

## Linear Regression

We'll train our algorithm on a training set, then test its performance on a test set. The train_test_split() function from scikit-learn will help us randomly select 20% of the rows for our test set, and 80% for our training set.


```python
# Train-test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)
```


```python
# Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate an instance
clf = LinearRegression()

# Fit the training data
clf.fit(X_train, y_train)

# Make predictions
y_predict = clf.predict(X_test)
```

### Error in Prediction

We'll use mean squared error (MSE), which is a common error metric. With MSE, we subtract the predictions from the actual values, square the results, and find the mean. Because the errors are squared, MSE penalizes errors further away from the actual value more than those close to the actual value. We have high error (rmse of ~50 upvotes) in predicting upvotes as we have used a very small data set. With larger training sets, this should decrease dramatically.


```python
mse = sum((y_predict - y_test) ** 2) / len(y_predict)
rmse = (mse)**0.5
print(rmse)
```

    51.5034780501


How to make better predictions?

- Using more data will ensure that the model will find more occurrences of the same features in the test and training sets, which will help the model make better predictions.

- Add "meta" features like headline length and average word length.

- Use a random forest, or another more powerful machine learning technique.

- Explore different thresholds for removing extraneous columns.
