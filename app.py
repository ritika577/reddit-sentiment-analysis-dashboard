import streamlit as st
import pandas as pd
from db_connection import get_db_connection
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# Query the database
def run_query(query):
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Streamlit app
st.title("Reddit Sentiment Analysis Dashboard")

query = """
SELECT post_title, comment, post_compound, comment_compound,
            post_pos_sentiment, post_neg_sentiment, post_neu_sentiment,
            comment_pos_sentiment, comment_neg_sentiment, comment_neu_sentiment, created_at
FROM reddit_sentiments
WHERE created_at >= NOW() - INTERVAL '30 days'
ORDER BY created_at DESC
;
"""
df = run_query(query)


st.write("Showing sample data:")
# Fill numeric columns with 0
df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(0)

# Fill string columns with 'Unknown'
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('Unknown')
df.drop_duplicates(inplace=True)

def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['post_sentiment_label'] = df['post_compound'].apply(classify_sentiment)
df['comment_sentiment_label'] = df['comment_compound'].apply(classify_sentiment)
df['created_at'] = pd.to_datetime(df['created_at'])




# Sidebar
st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to:", ["Overview", "Post Sentiment", "Comment Sentiment", "Comparison"])

# Overview
if view == "Overview":
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Sentiment Label Counts")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Unique Posts Analyzed", df['post_title'].nunique())

    with col2:
        st.metric("Unique Comments Analyzed", df['comment'].nunique())
    
    # Group comments by post_title and calculate mean compound sentiment
    topic_sentiment = df.groupby('post_title')['comment_compound'].mean().reset_index()
    top_positive = topic_sentiment.sort_values(by='comment_compound', ascending=False).head(10)
    top_negative = topic_sentiment.sort_values(by='comment_compound').head(10)

    st.subheader("🌟 What Topics Spark the Most Positivity or Negativity?")

    # Group and calculate average comment sentiment per post
    topic_sentiment = df.groupby('post_title')['comment_compound'].mean().reset_index()

    # Top 10 Positive
    top_positive = topic_sentiment.sort_values(by='comment_compound', ascending=False).head(10)

    # Top 10 Negative
    top_negative = topic_sentiment.sort_values(by='comment_compound').head(10)

    # -- Positive Topics Chart --
    st.markdown("### 💚 Top 10 Positive Topics (Based on Comment Sentiment)")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=top_positive,
        y='post_title',
        x='comment_compound',
        hue='post_title', 
        palette='Greens_d',
        legend=False,
        ax=ax1
    )
    ax1.set_title('💬 Topics with the Most Positive Comment Sentiment', fontsize=14)
    ax1.set_xlabel('Average Comment Sentiment')
    ax1.set_ylabel('')
    ax1.invert_yaxis()
    sns.despine()
    st.pyplot(fig1)

    # -- Negative Topics Chart --
    st.markdown("### 💔 Top 10 Negative Topics (Based on Comment Sentiment)")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=top_negative,
        y='post_title',
        x='comment_compound',
        hue='post_title', 
        palette='Reds_r',
        legend=False,
        ax=ax2
    )
    ax2.set_title('💬 Topics with the Most Negative Comment Sentiment', fontsize=14)
    ax2.set_xlabel('Average Comment Sentiment')
    ax2.set_ylabel('')
    ax2.invert_yaxis()
    sns.despine()
    st.pyplot(fig2)


    weekly_sentiment = df.set_index('created_at').resample('W').agg({
    'comment_compound': 'mean',
    'post_compound': 'mean'
    }).reset_index()


    st.subheader("📊 Interactive Trend: Sentiment Over Time")

    melted = weekly_sentiment.melt(id_vars='created_at', value_vars=['comment_compound', 'post_compound'],
                                    var_name='Type', value_name='Sentiment')

    line_chart = alt.Chart(melted).mark_line(point=True).encode(
        x='created_at:T',
        y='Sentiment:Q',
        color='Type:N',
        tooltip=['created_at:T', 'Type:N', 'Sentiment:Q']
    ).properties(
        width=700,
        height=400,
        title="📊 Weekly Sentiment Trend for Posts & Comments"
    )

    st.altair_chart(line_chart, use_container_width=True)



elif view == "Post Sentiment":
    st.subheader("📌 Post Sentiment Distribution")

    search_term = st.text_input("🔍 Search by post title")

    # Filter dataframe if something is typed
    filtered_df = df[df['post_title'].str.contains(search_term, case=False, na=False)] if search_term else df
    filtered_df = filtered_df.drop_duplicates(subset='post_title')
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='post_sentiment_label', order=['Positive', 'Neutral', 'Negative'], palette='Set2', ax=ax)
    st.pyplot(fig)

    st.dataframe(filtered_df[['post_title', 'post_sentiment_label']])

    weekly_sentiment = df.set_index('created_at').resample('W')['post_compound'].mean().reset_index()
    # st.dataframe(weekly_sentiment)

    st.subheader("📈 Average Weekly Sentiment Trend")

    fig, ax = plt.subplots()
    ax.plot(weekly_sentiment['created_at'], weekly_sentiment['post_compound'], marker='o', linestyle='-')
    ax.set_title('Average Comment Sentiment Score by Week')
    ax.set_xlabel('Week')
    ax.set_ylabel('Avg Compound Score')
    ax.grid(True)

    st.pyplot(fig)



elif view == "Comment Sentiment":
    st.subheader("💬 Comment Sentiment Distribution")

    search_term = st.text_input("🔍 Search by comment text")

    filtered_df = df[df['comment'].str.contains(search_term, case=False, na=False)] if search_term else df
    filtered_df = filtered_df.drop_duplicates(subset='comment')
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='comment_sentiment_label', order=['Positive', 'Neutral', 'Negative'], palette='Set1', ax=ax)
    st.pyplot(fig)

    st.dataframe(filtered_df[['comment', 'comment_sentiment_label']])


# Comparison
elif view == "Comparison":
    st.subheader("📊 Post vs Comment Sentiment Match")

    cross_tab = pd.crosstab(df['post_sentiment_label'], df['comment_sentiment_label'])

    fig, ax = plt.subplots()
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("This heatmap shows how post and comment sentiments align or differ.")


