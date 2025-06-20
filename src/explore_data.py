import pandas as pd
import json
import ast

def explore_twitter_dataset(data_path: str):
    """Explore the structure of the Twitter dataset."""
    
    print("🔍 Exploring Twitter Dataset Structure")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"📊 Dataset size: {len(df):,} rows × {len(df.columns)} columns")
    
    # Show column names
    print(f"\n📋 Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Basic info about key columns
    key_columns = ['id', 'text', 'createdAt', 'author', 'isReply', 'isQuote', 
                   'inReplyToUserId', 'retweetCount', 'likeCount']
    
    print(f"\n📈 Key Column Statistics:")
    for col in key_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            print(f"  {col:<20}: {non_null:,} non-null, {null_count:,} null")
    
    # Sample author field to understand structure
    print(f"\n👤 Sample Author Field:")
    sample_authors = df['author'].dropna().head(3)
    for i, author in enumerate(sample_authors, 1):
        print(f"\n  Sample {i}:")
        print(f"    Type: {type(author)}")
        print(f"    Content: {str(author)[:200]}...")
    
    # Sample tweet text
    print(f"\n💬 Sample Tweets:")
    sample_texts = df['text'].dropna().head(3)
    for i, text in enumerate(sample_texts, 1):
        print(f"\n  Tweet {i}: {str(text)[:100]}...")
    
    # Check interaction columns
    print(f"\n🔗 Interaction Information:")
    print(f"  Replies: {df['isReply'].sum() if 'isReply' in df.columns else 'N/A'}")
    print(f"  Quotes: {df['isQuote'].sum() if 'isQuote' in df.columns else 'N/A'}")
    print(f"  Has reply-to user: {df['inReplyToUserId'].notna().sum() if 'inReplyToUserId' in df.columns else 'N/A'}")
    
    # Date range
    if 'createdAt' in df.columns:
        dates = pd.to_datetime(df['createdAt'], errors='coerce')
        print(f"\n📅 Date Range:")
        print(f"  Earliest: {dates.min()}")
        print(f"  Latest: {dates.max()}")
        print(f"  Valid dates: {dates.notna().sum():,}")
    
    # Engagement metrics
    engagement_cols = ['retweetCount', 'replyCount', 'likeCount', 'quoteCount', 'viewCount']
    print(f"\n📊 Engagement Metrics:")
    for col in engagement_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            max_val = df[col].max()
            print(f"  {col:<15}: avg {mean_val:.1f}, max {max_val:,}")
    
    return df

def test_author_parsing(df: pd.DataFrame, n_samples: int = 5):
    """Test parsing author information."""
    print(f"\n🧪 Testing Author Parsing:")
    print("=" * 30)
    
    sample_authors = df['author'].dropna().head(n_samples)
    
    for i, author_str in enumerate(sample_authors, 1):
        print(f"\nSample {i}:")
        try:
            # Try different parsing methods
            if isinstance(author_str, str):
                # Method 1: JSON parsing
                try:
                    author_str_clean = author_str.replace("'", '"')
                    author_data = json.loads(author_str_clean)
                    print(f"  ✅ JSON parsing successful")
                    print(f"     User ID: {author_data.get('id', 'N/A')}")
                    print(f"     Username: {author_data.get('username', 'N/A')}")
                    print(f"     Followers: {author_data.get('followersCount', 'N/A')}")
                except json.JSONDecodeError:
                    # Method 2: AST parsing
                    try:
                        author_data = ast.literal_eval(author_str.replace('"', "'"))
                        print(f"  ✅ AST parsing successful")
                        print(f"     User ID: {author_data.get('id', 'N/A')}")
                        print(f"     Username: {author_data.get('username', 'N/A')}")
                    except:
                        print(f"  ❌ Both parsing methods failed")
                        print(f"     Raw: {str(author_str)[:100]}...")
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    # Explore the dataset
    df = explore_twitter_dataset("../data/tweetdata-en.csv")
    
    # Test author parsing
    test_author_parsing(df)
    
    # Test the updated preprocessor
    print(f"\n🔧 Testing Updated Preprocessor:")
    print("=" * 40)
    
    from preprocess import TwitterDataPreprocessor
    
    preprocessor = TwitterDataPreprocessor("../data/tweetdata-en.csv")
    users, interactions = preprocessor.process_data(filter_business=True)
    
    print(f"\n✅ Preprocessing Results:")
    print(f"  Users extracted: {len(users):,}")
    print(f"  Interactions extracted: {len(interactions):,}")
    
    if users:
        print(f"\n👤 Sample users:")
        for i, (user_id, user_info) in enumerate(list(users.items())[:3], 1):
            print(f"  {i}. @{user_info['username']} ({user_info['name']}) - {user_info['followers_count']} followers")
    
    if interactions:
        print(f"\n🔗 Sample interactions:")
        for i, interaction in enumerate(interactions[:3], 1):
            print(f"  {i}. {interaction['edge_type']}: {interaction['source_user_id']} -> {interaction['target_user_id']}")