import pandas as pd
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Set
import ast

class TwitterDataPreprocessor:
    """
    Preprocessor for Twitter data to extract user interactions and normalize metadata.
    Handles mentions, replies, retweets, and quotes to build temporal social graphs.
    Updated to work with the actual dataset column structure.
    """
    
    def __init__(self, data_path: str):
        """Initialize with path to CSV file."""
        self.data_path = data_path
        self.df = None
        self.users = {}
        self.interactions = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and return the tweet dataset."""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"✅ Loaded {len(self.df):,} tweets from {self.data_path}")
            print(f"📋 Columns available: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"❌ File not found: {self.data_path}")
            # Try alternative filenames
            alternative_paths = [
                self.data_path.replace('tweetdata-en.csv', 'tweetdata.csv'),
                self.data_path.replace('tweetdata.csv', 'tweetdata-en.csv')
            ]
            for alt_path in alternative_paths:
                try:
                    self.df = pd.read_csv(alt_path, encoding='utf-8')
                    print(f"✅ Found alternative file: {alt_path}")
                    print(f"✅ Loaded {len(self.df):,} tweets")
                    self.data_path = alt_path
                    return self.df
                except FileNotFoundError:
                    continue
            print(f"❌ No valid data file found")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean tweet text by removing extra whitespace and normalizing."""
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from tweet text."""
        if pd.isna(text):
            return []
        # Find all @username patterns
        mentions = re.findall(r'@(\w+)', str(text))
        return [mention.lower() for mention in mentions]
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from tweet text."""
        if pd.isna(text):
            return []
        # Find all #hashtag patterns
        hashtags = re.findall(r'#(\w+)', str(text))
        return [hashtag.lower() for hashtag in hashtags]
    
    def parse_datetime(self, date_str: str) -> datetime:
        """Parse datetime string to datetime object."""
        if pd.isna(date_str):
            return None
        try:
            # Try common Twitter datetime formats
            formats = [
                '%a %b %d %H:%M:%S %z %Y',  # Mon Apr 21 15:49:48 +0000 2025
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ'
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(str(date_str), fmt)
                except ValueError:
                    continue
            # If none work, try pandas
            return pd.to_datetime(date_str)
        except Exception:
            return None
    
    def parse_author_info(self, author_str: str) -> Dict:
        """Parse author information from the author field."""
        if pd.isna(author_str) or author_str == '':
            return {}
        
        try:
            # The author field appears to be a dictionary-like string
            if isinstance(author_str, str):
                # Replace single quotes with double quotes for JSON parsing
                author_str_clean = author_str.replace("'", '"')
                
                try:
                    # Try JSON parsing first
                    author_data = json.loads(author_str_clean)
                except json.JSONDecodeError:
                    try:
                        # Try ast.literal_eval for Python dict format
                        author_data = ast.literal_eval(author_str.replace('"', "'"))
                    except:
                        # Fallback to regex extraction
                        return self._extract_author_with_regex(author_str)
            else:
                author_data = author_str
            
            if isinstance(author_data, dict):
                # Extract information based on your actual data structure
                return {
                    'user_id': str(author_data.get('id', '')),
                    'username': str(author_data.get('userName', '')).lower(),
                    'name': author_data.get('name', ''),
                    'followers_count': int(author_data.get('followers', 0)),
                    'following_count': int(author_data.get('following', 0)),
                    'verified': bool(author_data.get('isVerified', False)),
                    'location': author_data.get('location', ''),
                    'description': author_data.get('description', '')
                }
        except Exception as e:
            # Fallback: extract basic info from string patterns
            return self._extract_author_with_regex(str(author_str))
        
        return {}
    
    def _extract_author_with_regex(self, author_str: str) -> Dict:
        """Fallback method to extract author info using regex."""
        try:
            # Extract key information using regex patterns based on your actual data structure
            user_id_match = re.search(r"'id':\s*'?(\d+)'?", str(author_str))
            username_match = re.search(r"'userName':\s*'([^']+)'", str(author_str))
            name_match = re.search(r"'name':\s*'([^']+)'", str(author_str))
            followers_match = re.search(r"'followers':\s*(\d+)", str(author_str))
            
            return {
                'user_id': user_id_match.group(1) if user_id_match else '',
                'username': username_match.group(1).lower() if username_match else '',
                'name': name_match.group(1) if name_match else '',
                'followers_count': int(followers_match.group(1)) if followers_match else 0,
                'following_count': 0,
                'verified': 'isVerified": True' in str(author_str),
                'location': '',
                'description': ''
            }
        except Exception:
            return {}
    
    def extract_user_info(self, row: pd.Series) -> Dict:
        """Extract user information from tweet row."""
        # Parse author information
        author_info = self.parse_author_info(row.get('author', ''))
        
        if not author_info.get('user_id'):
            return None
            
        # Clean and validate user info
        user_info = {
            'user_id': author_info['user_id'],
            'username': author_info['username'],
            'name': self.clean_text(author_info['name']),
            'followers_count': author_info['followers_count'],
            'following_count': author_info['following_count'],
            'verified': author_info['verified'],
            'location': self.clean_text(author_info['location']),
            'description': self.clean_text(author_info['description'])
        }
        return user_info
    
    def extract_interactions(self, row: pd.Series) -> List[Dict]:
        """Extract all interactions from a tweet row."""
        interactions = []
        
        # Get basic tweet info
        author_info = self.parse_author_info(row.get('author', ''))
        source_user = author_info.get('user_id', '') if author_info else ''
        tweet_id = str(row.get('id', ''))
        timestamp = self.parse_datetime(row.get('createdAt'))
        text = self.clean_text(row.get('text', ''))
        
        if not source_user or not timestamp:
            return interactions
        
        # Extract mentions from text
        mentions = self.extract_mentions(text)
        for mention in mentions:
            interactions.append({
                'source_user_id': source_user,
                'target_username': mention,
                'target_user_id': None,  # Will be resolved later
                'tweet_id': tweet_id,
                'timestamp': timestamp,
                'edge_type': 'mention',
                'weight': 1.0
            })
        
        # Extract replies
        reply_to_user_id = str(row.get('inReplyToUserId', ''))
        is_reply = row.get('isReply', False)
        
        if is_reply and reply_to_user_id and reply_to_user_id != 'nan' and reply_to_user_id != '':
            interactions.append({
                'source_user_id': source_user,
                'target_user_id': reply_to_user_id,
                'target_username': str(row.get('inReplyToUsername', '')).lower(),
                'tweet_id': tweet_id,
                'timestamp': timestamp,
                'edge_type': 'reply',
                'weight': 2.0  # Replies are stronger interactions
            })
        
        # Extract retweets (check retweeted_tweet field)
        retweeted_tweet = row.get('retweeted_tweet', '')
        if pd.notna(retweeted_tweet) and retweeted_tweet and retweeted_tweet != '':
            try:
                # Try to parse retweeted tweet info
                retweeted_info = self.parse_author_info(str(retweeted_tweet))
                if retweeted_info.get('user_id'):
                    interactions.append({
                        'source_user_id': source_user,
                        'target_user_id': retweeted_info['user_id'],
                        'target_username': retweeted_info.get('username', ''),
                        'tweet_id': tweet_id,
                        'timestamp': timestamp,
                        'edge_type': 'retweet',
                        'weight': 1.5
                    })
            except:
                pass
        
        # Extract quotes
        is_quote = row.get('isQuote', False)
        quoted_tweet = row.get('quoted_tweet', '')
        
        if is_quote and pd.notna(quoted_tweet) and quoted_tweet and quoted_tweet != '':
            try:
                # Try to parse quoted tweet info
                quoted_info = self.parse_author_info(str(quoted_tweet))
                if quoted_info.get('user_id'):
                    interactions.append({
                        'source_user_id': source_user,
                        'target_user_id': quoted_info['user_id'],
                        'target_username': quoted_info.get('username', ''),
                        'tweet_id': tweet_id,
                        'timestamp': timestamp,
                        'edge_type': 'quote',
                        'weight': 1.2
                    })
            except:
                pass
        
        return interactions
    
    def resolve_usernames_to_ids(self):
        """Resolve usernames in mentions to user IDs using available data."""
        # Create username to user_id mapping
        username_to_id = {}
        for user_id, user_info in self.users.items():
            username = user_info.get('username', '').lower()
            if username:
                username_to_id[username] = user_id
        
        # Update interactions to include user IDs for mentions
        for interaction in self.interactions:
            if interaction['edge_type'] == 'mention' and not interaction.get('target_user_id'):
                target_username = interaction.get('target_username', '').lower()
                if target_username in username_to_id:
                    interaction['target_user_id'] = username_to_id[target_username]

    def filter_business_related(self, business_keywords: List[str] = None) -> pd.DataFrame:
        """Filter tweets for business-related content."""
        if business_keywords is None:
            business_keywords = [
                'business', 'startup', 'entrepreneur', 'company', 'corporate',
                'finance', 'investment', 'marketing', 'sales', 'strategy',
                'innovation', 'leadership', 'management', 'revenue', 'profit',
                'economy', 'market', 'industry', 'growth', 'success', 'work',
                'job', 'career', 'professional', 'commercial', 'trade',
                'customer', 'client', 'service', 'product', 'brand', 'tech',
                'technology', 'digital', 'software', 'platform', 'solution'
            ]
        
        # Create pattern for business keywords
        pattern = '|'.join([f'\\b{keyword}\\b' for keyword in business_keywords])
        
        # Filter tweets containing business keywords
        business_mask = self.df['text'].str.contains(pattern, case=False, na=False)
        business_df = self.df[business_mask].copy()
        
        print(f"🔍 Filtered to {len(business_df):,} business-related tweets from {len(self.df):,} total tweets")
        return business_df
    
    def process_data(self, filter_business: bool = True) -> Tuple[Dict, List[Dict]]:
        """Main processing method to extract users and interactions."""
        if self.df is None:
            self.load_data()
        
        if self.df is None:
            return {}, []
        
        # Filter for business-related content if requested
        if filter_business:
            processed_df = self.filter_business_related()
        else:
            processed_df = self.df.copy()
        
        print("📤 Extracting user information...")
        # Extract users
        user_count = 0
        for idx, row in processed_df.iterrows():
            user_info = self.extract_user_info(row)
            if user_info and user_info['user_id']:
                # Update user info if we have more complete data
                if user_info['user_id'] not in self.users:
                    self.users[user_info['user_id']] = user_info
                    user_count += 1
                else:
                    # Update with non-empty values
                    for key, value in user_info.items():
                        if value and (not self.users[user_info['user_id']].get(key) or 
                                    self.users[user_info['user_id']][key] == ''):
                            self.users[user_info['user_id']][key] = value
        
        print(f"👥 Extracted {user_count} new users, total: {len(self.users)}")
        
        print("🔗 Extracting interactions...")
        # Extract interactions
        interaction_count = 0
        for idx, row in processed_df.iterrows():
            tweet_interactions = self.extract_interactions(row)
            self.interactions.extend(tweet_interactions)
            interaction_count += len(tweet_interactions)
        
        print(f"📊 Extracted {interaction_count} total interactions")
        
        # Resolve usernames to user IDs
        self.resolve_usernames_to_ids()
        
        # Filter interactions with valid target users
        valid_interactions = []
        for interaction in self.interactions:
            if interaction['target_user_id'] and interaction['target_user_id'] in self.users:
                valid_interactions.append(interaction)
        
        removed_interactions = len(self.interactions) - len(valid_interactions)
        self.interactions = valid_interactions
        
        print(f"✅ Processing complete:")
        print(f"   Users: {len(self.users):,}")
        print(f"   Valid interactions: {len(self.interactions):,}")
        print(f"   Removed invalid interactions: {removed_interactions:,}")
        
        # Print interaction type breakdown
        if self.interactions:
            interaction_types = {}
            for interaction in self.interactions:
                edge_type = interaction['edge_type']
                interaction_types[edge_type] = interaction_types.get(edge_type, 0) + 1
            
            print(f"\n📈 Interaction type breakdown:")
            for edge_type, count in sorted(interaction_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.interactions)) * 100
                print(f"   {edge_type.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        return self.users, self.interactions
    
    def get_processed_tweets(self) -> pd.DataFrame:
        """Return processed tweets with additional metadata."""
        if self.df is None:
            return pd.DataFrame()
        
        processed_df = self.df.copy()
        processed_df['text_clean'] = processed_df['text'].apply(self.clean_text)
        processed_df['mentions'] = processed_df['text'].apply(self.extract_mentions)
        processed_df['hashtags'] = processed_df['text'].apply(self.extract_hashtags)
        processed_df['created_at_parsed'] = processed_df['createdAt'].apply(self.parse_datetime)
        
        # Extract user info for each tweet
        user_ids = []
        usernames = []
        for idx, row in processed_df.iterrows():
            author_info = self.parse_author_info(row.get('author', ''))
            user_ids.append(author_info.get('user_id', ''))
            usernames.append(author_info.get('username', ''))
        
        processed_df['user_id'] = user_ids
        processed_df['username'] = usernames
        processed_df['created_at'] = processed_df['createdAt']  # Standardize column name
        
        return processed_df
    
    def save_processed_data(self, users_path: str, interactions_path: str):
        """Save processed users and interactions to files."""
        try:
            # Save users
            if self.users:
                users_df = pd.DataFrame.from_dict(self.users, orient='index')
                users_df.to_csv(users_path, index=False)
                print(f"💾 Saved {len(users_df)} users to {users_path}")
            
            # Save interactions
            if self.interactions:
                interactions_df = pd.DataFrame(self.interactions)
                interactions_df.to_csv(interactions_path, index=False)
                print(f"💾 Saved {len(interactions_df)} interactions to {interactions_path}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")

if __name__ == "__main__":
    # Example usage
    preprocessor = TwitterDataPreprocessor("../data/tweetdata-en.csv")
    users, interactions = preprocessor.process_data(filter_business=True)
    
    # Save processed data
    preprocessor.save_processed_data(
        "../data/processed_users.csv",
        "../data/processed_interactions.csv"
    )