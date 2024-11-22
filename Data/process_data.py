import json
import re
import unidecode
import nltk
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

weights = {
    "user_count": 0.10, # Indicates wide-spread high interest
    "main_post_likes": 0.2, # Indicates wide-spread medium interest
    "mean_comment_likes": 0.15, # Indicates wide-spread medium interest
    "stddev_comment_likes": 0.05, # Indicates Polarization
    "code_references": 0.2, # Indicates wide-spread medium-high interest
    "replies": 0.1, # Indicates wide-spread high interest
    "views": 0.15, # Indicates wide-spread low interest
    'adjusted_time_gap': 0.3  # Indicated high interest
}

def get_posts():
    with open('posts.json', "r") as f:
        posts = json.load(f)
    
    return posts

def get_rules():
    with open('rules.json', "r") as f:
        rules = json.load(f)
    
    return rules

def target_information(series):
    print("INFO: Distribution of target...")
    print(f"INFO: Mean of target: {np.mean(series)}")
    print(f"INFO: Median of target: {np.median(series)}")
    print(f"INFO: Std. Dev. of target: {np.std(series)}")

    counts, bins = np.histogram(series)
    plt.stairs(counts, bins)
    plt.title('Controversy Score Distribution')
    plt.xlabel('Controversy Score')
    plt.ylabel('Frequency')
    plt.show()

def count_and_remove_codeblocks(text):
    # Remove codeblock indicators and the code between
    code_start_indices = list(re.finditer(r'CODEBLOCK_START', text))
    code_end_indices = list(re.finditer(r'CODEBLOCK_END', text))

    # Number of code references
    code_ref_count = len(code_start_indices)

    # Delete code from descriptions
    for start, end in zip(reversed(code_start_indices), reversed(code_end_indices)):
        text = text[:start.start()] + text[end.end():]

    # Delete other well-defined code-related headers
    code_tells = [r"(\s)Noncompliant code example(\s)", r"(\s)Compliant solution(\s)", r"(\s)Exceptions(\s)"]
    for tell in code_tells:
        text = re.sub(tell, ' ', text)

    return text, code_ref_count

def clean_description(text):
    # Remove newline characters
    text = re.sub(r'\\n', ' ', text)
    # Decode unicode
    text = unidecode.unidecode(text)
    # Reduce multiple spaces to single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-text
    text = re.sub("[^\w\s]", " ", text) 
    # Remove stop words
    stops = set(stopwords.words("english")) 
    words = word_tokenize(text.lower())
    meaningful_words = [w for w in words if not w in stops]   
    # Stem words
    snow_stemmer = SnowballStemmer(language='english')
    words = ([snow_stemmer.stem(w) for w in meaningful_words])
    text = " ".join(words)

    return text

def main():
    posts = get_posts()
    rules = get_rules()

    rules_df = pd.DataFrame.from_dict(rules)

    # Map combined rule codes to corresponding rule entries
    rules_df['rule'] = rules_df.apply(lambda row: f"{row['language']}:{row['code']}", axis=1)

    # Clean rule descriptions
    rules_df[['description', 'codeblock_count']] = rules_df['description'].apply(
        lambda desc: pd.Series(count_and_remove_codeblocks(desc))
    )
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    rules_df['description'] = rules_df['description'].apply(clean_description)

    posts_df = pd.DataFrame.from_dict(posts)
    # Remove duplicates (occurs when scraper had to be re-run from a checkpoint due to a network error)
    posts_df.drop_duplicates('id')

    numeric_cols = ["link_count", "user_count", "main_post_likes", "code_references", "mean_comment_likes", "stddev_comment_likes", "comment_likes", "replies", "views"]
    for col in numeric_cols: 
        posts_df[col] = pd.to_numeric(posts_df[col])

    # Calculate the time gap in days
    posts_df['created'] = pd.to_datetime(posts_df['datetimes'].apply(lambda x: x['created']))
    posts_df['latest'] = pd.to_datetime(posts_df['datetimes'].apply(lambda x: x['latest']))
    posts_df['time_gap_days'] = (posts_df['latest'] - posts_df['created']).dt.days
    posts_df['age_days'] = (datetime.now() - posts_df['created']).dt.days

    # Decay constant
    decay_constant = 365 * 2  # Decay over about 2 years

    # Calculate the decay factor based on age
    posts_df['decay_factor'] = np.exp(-posts_df['age_days'] / decay_constant)

    # Apply decay to the time gap (so that older posts, with more time to seed themselves in SEO, don't skew the complexity score)
    # posts_df['adjusted_time_gap'] = posts_df['time_gap_days'] * posts_df['decay_factor']
    posts_df['adjusted_time_gap'] = posts_df['time_gap_days'] * posts_df['decay_factor']

    # Calculate and normalize complexity scores
    scaler = MinMaxScaler()
    numeric_cols.append('adjusted_time_gap')
    posts_df[numeric_cols] = scaler.fit_transform(posts_df[numeric_cols])

    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    posts_df['controversy_score'] = (
        posts_df['user_count'] * normalized_weights['user_count'] +
        posts_df['main_post_likes'] * normalized_weights['main_post_likes'] +
        posts_df['mean_comment_likes'] * normalized_weights['mean_comment_likes'] +
        posts_df['stddev_comment_likes'] * normalized_weights['stddev_comment_likes'] +
        posts_df['code_references'] * normalized_weights['code_references'] +
        posts_df['replies'] * normalized_weights['replies'] +
        posts_df['views'] * normalized_weights['views'] + 
        posts_df['adjusted_time_gap'] * normalized_weights['adjusted_time_gap']
    )

    # Match post controversy scores to their referenced rules
    posts_df['rule'] = posts_df['rule']
    controversy_per_rule = posts_df.groupby('rule')['controversy_score'].apply(list).reset_index()
    
    # Merge the aggregated scores back into rules_df
    rules_df = rules_df.merge(controversy_per_rule, on='rule', how='left')
    rules_df['controversy_score'] = rules_df['controversy_score'].apply(lambda d: d if isinstance(d, list) else [0]) # Missing scores (no controversy found)
    missing_count = rules_df['controversy_score'].apply(lambda x: x == [0]).sum()
    print(f"INFO: {missing_count} Sonar rules did not have any controversy scores associated with them out of {len(rules_df)}.")

    # Controversy score (averaged) distributions of those not equal to 0
    # Keep rows where controversy_score != [0]
    non_zero_controversy = rules_df.loc[rules_df['controversy_score'].apply(lambda x: x != [0])]
    print(f"INFO: Learning will utilize {len(non_zero_controversy.index)} rules with a corresponding controversy score out of {len(rules_df)} total rules.")
    controversy_scores_means = non_zero_controversy['controversy_score'].apply(np.mean).values
    target_information(controversy_scores_means)

    # Save combined data
    cleanData = 'posts-cleaned.json'

    with open(cleanData, 'w') as file:
        file.write(rules_df.to_json(orient = "records"))

    return

main()