from db_connection import get_db_connection
import praw
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
# Load environment variables from .env
# load_dotenv()
conn=get_db_connection()

def get_sentiment(text):
    return analyzer.polarity_scores(text)

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)

# Choose a subreddit
subreddit = reddit.subreddit("india")

data = []

for post in subreddit.hot(limit=200):
    post.comments.replace_more(limit=0)
    post_sent = get_sentiment(post.title + " " + post.selftext)
    # print(post_sent)

    for comment in post.comments[:5]:
        comment_sent = get_sentiment(comment.body)
        data.append({
            'post_title': post.title,
            'comment': comment.body,
            'post_compound': post_sent['compound'],
            'comment_compound': comment_sent['compound'],
            'post_pos_sent' : post_sent['pos'],
            'post_neg_sent' : post_sent['neg'],
            'post_neu_sent' : post_sent['neu'],
            'comment_pos_sent' : comment_sent['pos'],
            'comment_neg_sent' : comment_sent['neg'],
            'comment_neu_sent' : comment_sent['neu']
            
        })




if conn:
    cursor = conn.cursor()

    query = """
        INSERT INTO reddit_sentiments (
            post_title, comment, post_compound, comment_compound,
            post_pos_sentiment, post_neg_sentiment, post_neu_sentiment,
            comment_pos_sentiment, comment_neg_sentiment, comment_neu_sentiment,
            created_at, updated_at
        ) VALUES (
            %(post_title)s, %(comment)s, %(post_compound)s, %(comment_compound)s,
            %(post_pos_sent)s, %(post_neg_sent)s, %(post_neu_sent)s,
            %(comment_pos_sent)s, %(comment_neg_sent)s, %(comment_neu_sent)s,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        );
    """

    try:
        execute_batch(cursor, query, data, page_size=100)  # batch insert in chunks of 100
        conn.commit()
        print(f"✅ {len(data)} rows inserted successfully.")
    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        conn.rollback()

    cursor.close()
    conn.close()
else:
    print("⚠️ Connection failed.")
