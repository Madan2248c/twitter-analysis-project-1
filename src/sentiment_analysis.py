import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class TwitterSentimentAnalyzer:
    """
    Analyzes sentiment of tweets using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Provides temporal sentiment analysis and opinion tracking without machine learning.
    """
    
    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_data = []
        self.temporal_sentiment = {}
        
    def preprocess_tweet_text(self, text: str) -> str:
        """Preprocess tweet text for better sentiment analysis."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Preserve sentiment-relevant elements
        # Convert common internet slang and emoticons
        text = re.sub(r':\)', ' :) ', text)  # Happy emoticon
        text = re.sub(r':\(', ' :( ', text)  # Sad emoticon
        text = re.sub(r':D', ' :D ', text)   # Very happy
        text = re.sub(r'<3', ' <3 ', text)   # Heart
        text = re.sub(r'</3', ' </3 ', text) # Broken heart
        
        # Handle repeated characters (enthusiasm indicators)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Reduce repeated chars to max 2
        
        # Handle URLs (replace with neutral term)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
        
        # Handle mentions and hashtags (keep the # and @ as they can indicate sentiment)
        # Don't remove them as they're sentiment-relevant
        
        return text.strip()
    
    def analyze_tweet_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a single tweet."""
        processed_text = self.preprocess_tweet_text(text)
        
        if not processed_text:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'sentiment_label': 'neutral'
            }
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(processed_text)
        
        # Determine sentiment label based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'sentiment_label': label
        }
    
    def analyze_dataset_sentiment(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Analyze sentiment for entire dataset."""
        sentiment_results = []
        
        print(f"Analyzing sentiment for {len(df)} tweets...")
        
        for idx, row in df.iterrows():
            tweet_text = row.get(text_column, '')
            sentiment = self.analyze_tweet_sentiment(tweet_text)
            
            # Combine with original tweet data
            result = {
                'tweet_id': row.get('id', ''),
                'user_id': row.get('user_id', ''),
                'username': row.get('username', ''),
                'created_at': row.get('created_at', ''),
                'text': tweet_text,
                'text_processed': self.preprocess_tweet_text(tweet_text),
                **sentiment
            }
            
            sentiment_results.append(result)
        
        sentiment_df = pd.DataFrame(sentiment_results)
        self.sentiment_data = sentiment_results
        
        # Print summary statistics
        self._print_sentiment_summary(sentiment_df)
        
        return sentiment_df
    
    def _print_sentiment_summary(self, sentiment_df: pd.DataFrame):
        """Print summary of sentiment analysis results."""
        total_tweets = len(sentiment_df)
        
        if total_tweets == 0:
            print("No tweets to analyze.")
            return
        
        # Count by sentiment label
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()
        
        print(f"\nSentiment Analysis Summary:")
        print(f"Total tweets analyzed: {total_tweets}")
        print(f"Sentiment distribution:")
        for label, count in sentiment_counts.items():
            percentage = (count / total_tweets) * 100
            print(f"  {label.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Average scores
        avg_compound = sentiment_df['compound'].mean()
        avg_positive = sentiment_df['positive'].mean()
        avg_negative = sentiment_df['negative'].mean()
        avg_neutral = sentiment_df['neutral'].mean()
        
        print(f"\nAverage sentiment scores:")
        print(f"  Compound: {avg_compound:.3f}")
        print(f"  Positive: {avg_positive:.3f}")
        print(f"  Negative: {avg_negative:.3f}")
        print(f"  Neutral: {avg_neutral:.3f}")
    
    def analyze_temporal_sentiment(self, sentiment_df: pd.DataFrame, interval: str = 'daily') -> pd.DataFrame:
        """Analyze how sentiment changes over time."""
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Parse datetime
        sentiment_df['created_at_parsed'] = pd.to_datetime(sentiment_df['created_at'], errors='coerce')
        
        # Remove rows with invalid dates
        valid_dates = sentiment_df.dropna(subset=['created_at_parsed'])
        
        if valid_dates.empty:
            print("No valid timestamps found for temporal analysis.")
            return pd.DataFrame()
        
        # Group by time interval
        if interval == 'daily':
            valid_dates['time_period'] = valid_dates['created_at_parsed'].dt.date
        elif interval == 'weekly':
            valid_dates['time_period'] = valid_dates['created_at_parsed'].dt.to_period('W')
        elif interval == 'hourly':
            valid_dates['time_period'] = valid_dates['created_at_parsed'].dt.floor('H')
        else:
            valid_dates['time_period'] = valid_dates['created_at_parsed'].dt.date
        
        # Calculate temporal sentiment metrics
        temporal_groups = valid_dates.groupby('time_period')
        
        temporal_sentiment = []
        for period, group in temporal_groups:
            total_tweets = len(group)
            
            # Calculate averages
            avg_compound = group['compound'].mean()
            avg_positive = group['positive'].mean()
            avg_negative = group['negative'].mean()
            avg_neutral = group['neutral'].mean()
            
            # Count sentiment labels
            sentiment_counts = group['sentiment_label'].value_counts()
            positive_count = sentiment_counts.get('positive', 0)
            negative_count = sentiment_counts.get('negative', 0)
            neutral_count = sentiment_counts.get('neutral', 0)
            
            # Calculate percentages
            positive_pct = (positive_count / total_tweets) * 100
            negative_pct = (negative_count / total_tweets) * 100
            neutral_pct = (neutral_count / total_tweets) * 100
            
            # Determine dominant sentiment
            max_count = max(positive_count, negative_count, neutral_count)
            if max_count == positive_count:
                dominant_sentiment = 'positive'
            elif max_count == negative_count:
                dominant_sentiment = 'negative'
            else:
                dominant_sentiment = 'neutral'
            
            temporal_sentiment.append({
                'time_period': period,
                'total_tweets': total_tweets,
                'avg_compound': avg_compound,
                'avg_positive': avg_positive,
                'avg_negative': avg_negative,
                'avg_neutral': avg_neutral,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'positive_pct': positive_pct,
                'negative_pct': negative_pct,
                'neutral_pct': neutral_pct,
                'dominant_sentiment': dominant_sentiment,
                'sentiment_volatility': group['compound'].std()  # Measure of sentiment variation
            })
        
        temporal_df = pd.DataFrame(temporal_sentiment)
        self.temporal_sentiment = temporal_df
        
        print(f"Created temporal sentiment analysis for {len(temporal_df)} time periods")
        return temporal_df
    
    def analyze_user_sentiment_patterns(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment patterns by user."""
        if sentiment_df.empty:
            return pd.DataFrame()
        
        user_groups = sentiment_df.groupby('user_id')
        user_sentiment_patterns = []
        
        for user_id, group in user_groups:
            if len(group) < 2:  # Skip users with only one tweet
                continue
            
            total_tweets = len(group)
            
            # Calculate user's average sentiment
            avg_compound = group['compound'].mean()
            std_compound = group['compound'].std()
            
            # Count sentiment distribution
            sentiment_counts = group['sentiment_label'].value_counts()
            positive_count = sentiment_counts.get('positive', 0)
            negative_count = sentiment_counts.get('negative', 0)
            neutral_count = sentiment_counts.get('neutral', 0)
            
            # Calculate consistency (how often user expresses same sentiment)
            max_sentiment_count = max(positive_count, negative_count, neutral_count)
            consistency_score = max_sentiment_count / total_tweets
            
            # Determine user's dominant sentiment
            if positive_count >= negative_count and positive_count >= neutral_count:
                dominant_sentiment = 'positive'
            elif negative_count >= neutral_count:
                dominant_sentiment = 'negative'
            else:
                dominant_sentiment = 'neutral'
            
            # Get user info
            user_info = group.iloc[0]
            
            user_sentiment_patterns.append({
                'user_id': user_id,
                'username': user_info.get('username', ''),
                'total_tweets': total_tweets,
                'avg_compound': avg_compound,
                'sentiment_volatility': std_compound,
                'dominant_sentiment': dominant_sentiment,
                'consistency_score': consistency_score,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'positive_pct': (positive_count / total_tweets) * 100,
                'negative_pct': (negative_count / total_tweets) * 100,
                'neutral_pct': (neutral_count / total_tweets) * 100
            })
        
        user_patterns_df = pd.DataFrame(user_sentiment_patterns)
        
        print(f"Analyzed sentiment patterns for {len(user_patterns_df)} users")
        return user_patterns_df
    
    def get_sentiment_extremes(self, sentiment_df: pd.DataFrame, top_k: int = 10) -> Dict[str, pd.DataFrame]:
        """Get most positive and negative tweets."""
        if sentiment_df.empty:
            return {'most_positive': pd.DataFrame(), 'most_negative': pd.DataFrame()}
        
        # Sort by compound score
        most_positive = sentiment_df.nlargest(top_k, 'compound')[
            ['tweet_id', 'username', 'text', 'compound', 'sentiment_label', 'created_at']
        ]
        
        most_negative = sentiment_df.nsmallest(top_k, 'compound')[
            ['tweet_id', 'username', 'text', 'compound', 'sentiment_label', 'created_at']
        ]
        
        return {
            'most_positive': most_positive,
            'most_negative': most_negative
        }
    
    def analyze_hashtag_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment associated with different hashtags."""
        if sentiment_df.empty:
            return pd.DataFrame()
        
        hashtag_sentiment = []
        
        for idx, row in sentiment_df.iterrows():
            text = row.get('text', '')
            hashtags = re.findall(r'#(\w+)', text.lower())
            
            for hashtag in hashtags:
                hashtag_sentiment.append({
                    'hashtag': hashtag,
                    'compound': row['compound'],
                    'sentiment_label': row['sentiment_label']
                })
        
        if not hashtag_sentiment:
            return pd.DataFrame()
        
        hashtag_df = pd.DataFrame(hashtag_sentiment)
        
        # Aggregate by hashtag
        hashtag_analysis = hashtag_df.groupby('hashtag').agg({
            'compound': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral'
        }).reset_index()
        
        # Flatten column names
        hashtag_analysis.columns = ['hashtag', 'avg_compound', 'sentiment_volatility', 'mention_count', 'dominant_sentiment']
        
        # Filter hashtags with at least 3 mentions
        hashtag_analysis = hashtag_analysis[hashtag_analysis['mention_count'] >= 3]
        
        return hashtag_analysis.sort_values('mention_count', ascending=False)
    
    def save_sentiment_data(self, sentiment_df: pd.DataFrame, output_path: str):
        """Save sentiment analysis results."""
        sentiment_df.to_csv(output_path, index=False)
        print(f"Saved sentiment analysis to {output_path}")
    
    def load_sentiment_data(self, input_path: str) -> pd.DataFrame:
        """Load previously saved sentiment analysis results."""
        return pd.read_csv(input_path)

if __name__ == "__main__":
    # Example usage
    from preprocess import TwitterDataPreprocessor
    
    # Load and preprocess data
    preprocessor = TwitterDataPreprocessor("../data/tweetdata-en.csv")
    preprocessor.load_data()
    
    # Get processed tweets (business-related)
    business_tweets = preprocessor.filter_business_related()
    
    # Initialize sentiment analyzer
    sentiment_analyzer = TwitterSentimentAnalyzer()
    
    # Analyze sentiment
    sentiment_df = sentiment_analyzer.analyze_dataset_sentiment(business_tweets)
    
    # Temporal sentiment analysis
    temporal_sentiment = sentiment_analyzer.analyze_temporal_sentiment(sentiment_df, interval='daily')
    
    # User sentiment patterns
    user_patterns = sentiment_analyzer.analyze_user_sentiment_patterns(sentiment_df)
    
    # Get sentiment extremes
    extremes = sentiment_analyzer.get_sentiment_extremes(sentiment_df, top_k=5)
    
    print("\nMost Positive Tweets:")
    for idx, row in extremes['most_positive'].iterrows():
        print(f"Score: {row['compound']:.3f} - @{row['username']}: {row['text'][:100]}...")
    
    print("\nMost Negative Tweets:")
    for idx, row in extremes['most_negative'].iterrows():
        print(f"Score: {row['compound']:.3f} - @{row['username']}: {row['text'][:100]}...")
    
    # Hashtag sentiment analysis
    hashtag_sentiment = sentiment_analyzer.analyze_hashtag_sentiment(sentiment_df)
    
    if not hashtag_sentiment.empty:
        print(f"\nTop hashtags by sentiment:")
        print(hashtag_sentiment.head(10).to_string(index=False))
    
    # Save results
    sentiment_analyzer.save_sentiment_data(sentiment_df, "../data/sentiment_analysis.csv")