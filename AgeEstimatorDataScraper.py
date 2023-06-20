import praw
import pandas as pd
import re
import time
import os
import prawcore

# initialize praw with your own Reddit app credentials
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
    username="",
    password="",
)

# Load existing data or initialize an empty DataFrame
try:
    df = pd.read_csv('outputComplete2.csv')
except (FileNotFoundError, pd.errors.EmptyDataError):  # File does not exist or is empty
    df = pd.DataFrame(columns=['Username', 'Age', 'CommentAge', 'CommentId'])

# Function to process comments
def process_comments(post_id, pattern, is_match=False):
    post = reddit.submission(id=post_id)
    post.comments.replace_more(limit=0)  # this line expands the comment tree
    for comment in post.comments.list():
        # check if comment has an author
        if comment.author is not None:
            # check if author name is already in the DataFrame
            if comment.author.name not in df['Username'].values:
                # search for pattern in the comment
                match = pattern.findall(comment.body) if not is_match else pattern.match(comment.body)
                # add username and number to the list only if condition is met
                if (is_match and match is not None) or (not is_match and len(match) == 1):
                    # if match, extract age and calculate comment age
                    age = match.group().rstrip(',y, yo') if is_match else match[0]
                    comment_age = (time.time() - comment.created_utc) / (60 * 60 * 24 * 365)
                    new_data = {'Username': comment.author.name, 'Age': age, 'CommentAge': comment_age, 'CommentId': comment.id}
                    # Create a new DataFrame for the new data and write it to the CSV file
                    new_df = pd.DataFrame([new_data])
                    new_df.to_csv('outputComplete2.csv', mode='a', header=not os.path.exists('outputComplete2.csv'), index=False)
                    # Print the new data that has been added
                    print(f"New row added: {new_data}")
                    # Pause and wait for user to press Enter
                    input("Press Enter to continue...")

# regex pattern to match two digit number
pattern = re.compile(r'\b\d{2}\b')

# specify the post ids and patterns
posts_patterns = [
    ('rx5dqy', pattern, False),
    ('6yl74k', pattern, False),
    ('vc65ql', re.compile(r'^\d{2}(,|y|yo)'), True)
]

# Process comments for each post
for post_id, pattern, is_match in posts_patterns:
    process_comments(post_id, pattern, is_match)

print(df)

# Load existing data
try:
    df = pd.read_csv('outputComplete2.csv')
except (FileNotFoundError, pd.errors.EmptyDataError):  # File does not exist or is empty
    print("No data found!")
    exit()

# add new columns for UserComments and CommentIds
if 'UserComments' not in df.columns:
    df['UserComments'] = ""
if 'CommentIds' not in df.columns:
    df['CommentIds'] = ""
if 'AvgCommentAge' not in df.columns:
    df['AvgCommentAge'] = None

# iterate over the users
for index, row in df.iterrows():
    username = row['Username']
    user = reddit.redditor(username)
    comment_counter = 0
    first_comment_time = None

    comment_ids = set()
    if pd.notnull(row['CommentIds']):
        comment_ids = set(row['CommentIds'].split("\n"))

    # add CommentId to CommentIds
    if pd.notnull(row['CommentId']):
        comment_ids.add(row['CommentId'])

    comments = []
    if pd.notnull(row['UserComments']):
        comments = row['UserComments'].split("\n")

    comment_age_sum = 0
    # get user's 30 recent comments
    try:
        for comment in user.comments.new(limit=60):
            if first_comment_time is None:
                first_comment_time = comment.created_utc
            comment_age_difference = first_comment_time - comment.created_utc
            if comment.id not in comment_ids and comment_age_difference <= 60 * 60 * 24 * 365:
                comment_age_years = comment_age_difference / (60 * 60 * 24 * 365)
                comments.append(comment.body.replace("\n", " "))  # replace new lines in comment body to prevent csv issues
                comment_ids.add(comment.id)
                comment_counter += 1
                comment_age_sum += comment_age_years
                print(f"Added comment {comment.id} from user {username}")
                if comment_counter == 60:
                    break
    except prawcore.exceptions.Forbidden:
        print(f"Cannot fetch comments for user {username} due to 403 Forbidden error")
        continue

    df.at[index, 'UserComments'] = "\n".join(comments)
    df.at[index, 'CommentIds'] = "\n".join(list(comment_ids))
    if comment_counter > 0:
        df.at[index, 'AvgCommentAge'] = comment_age_sum / comment_counter

# save the updated data frame to the csv file
df.to_csv('outputComplete2.csv', index=False)

print(df)
