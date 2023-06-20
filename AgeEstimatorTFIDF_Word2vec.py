import pandas as pd
import numpy as np
import re
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import FunctionTransformer
from gensim.models import Word2Vec

# Load the data
df = pd.read_csv("outputComplete.csv")

# Remove any rows with missing values
df.dropna(inplace=True)

# Calculate the age of the user when commenting
df['AgeWhenCommenting'] = df['Age'] + df['CommentAge'] - df['AvgCommentAge']

# Function to count emojis in text
def count_emojis(text):
    # Define emoji pattern
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    # Return count of emojis
    return len(emoji_pattern.findall(text))

# Function to count words in text
def count_words(text):
    return len(text.split())

# Apply count functions to each user comment
df['EmojiCount'] = df['UserComments'].apply(count_emojis)
df['WordCount'] = df['UserComments'].apply(count_words)

# Define input features and target variable
X = df[['UserComments', 'EmojiCount', 'WordCount']]
y = df['AgeWhenCommenting']

# TF-IDF Model
print("TF-IDF results:")
# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'), 'UserComments'),  # Convert comments to TF-IDF
        ('emoji', FunctionTransformer(np.reshape, kw_args={'newshape':(-1,1)}), 'EmojiCount'),  # Keep emoji count as is
        ('words', FunctionTransformer(np.reshape, kw_args={'newshape':(-1,1)}), 'WordCount')])  # Keep word count as is

# Create a pipeline that combines preprocessing and model training
model = make_pipeline(preprocessor, LinearRegression())

# Train model and make predictions using cross-validation
predictions = cross_val_predict(model, X, y, cv=5)

# Calculate absolute error
diff = abs(y - predictions)

# Calculate proportion of predictions within 10 years of actual age
accuracy_within_10 = (diff <= 10).mean()

print('Proportion of predictions within 10 years of the true age:', accuracy_within_10)

# Word2Vec Model
print("Word2Vec results:")
# Prepare comments for word2vec
comments = df['UserComments'].apply(lambda x: [comment.split() for comment in x.split('\n')])
sentences = [sentence for user_comments in comments for sentence in user_comments]

# Train word2vec model
model = Word2Vec(sentences, min_count=1)

# Create word embeddings for each user's comments
user_comment_vectors = []
for user_comments in comments:
    comment_vectors = []
    for comment in user_comments:
        word_vectors = [model.wv[word] for word in comment if word in model.wv]
        if word_vectors:
            comment_vectors.append(np.mean(word_vectors, axis=0))
    if comment_vectors:
        user_comment_vectors.append(np.mean(comment_vectors, axis=0))
    else:
        user_comment_vectors.append(np.zeros(model.vector_size))

# Combine word embeddings with additional features (emoji count and word count)
X = np.concatenate((np.array(user_comment_vectors), df[['EmojiCount', 'WordCount']].to_numpy()), axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate absolute error
diff = abs(y_test - y_pred)

# Calculate proportion of predictions within 10 years of actual age
accuracy_within_10 = (diff <= 10).mean()

print('Proportion of predictions within 10 years of the true age:', accuracy_within_10)
