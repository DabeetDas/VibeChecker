# %%
# Importing libraries

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# %%
# Loading the csv file
df = pd.read_csv('Merged_data.csv')

# %%
# EDA 1 (Checking how many posts have images)

import matplotlib.pyplot as plt
import seaborn as sns

df['has_image'] = df['image_url'].notna()

plt.figure(figsize=(6,4))
sns.countplot(x='has_image', data=df, palette=['red', 'green'])
plt.title("Do the posts have images??")
plt.xticks([0,1], ['No Image', 'Has Image'])
plt.show()

# %%
# detailed view
print(df.head())

# %%
# GENERTING EMBEDDINGS USING CLIP

import numpy as np
import torch
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
from tqdm.notebook import tqdm
from transformers import CLIPProcessor, CLIPModel


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

final_image_embeddings = []
final_text_embeddings = []
valid_indices = []

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

batch_size = 16
urls = df['image_url'].tolist()
titles = df['title'].tolist()

for i in tqdm(range(0, len(urls), batch_size)):
    batch_urls = urls[i : i + batch_size]
    batch_titles = titles[i : i + batch_size]
    
    current_batch_images = []
    current_batch_texts = []
    current_indices = []
    
    for j, url in enumerate(batch_urls):
        try:
            if pd.isnull(url) or str(url).strip() == "":
                continue
            response = requests.get(url, headers=headers, timeout=3)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                image = image.resize((224, 224))
                
                current_batch_images.append(image)
                current_batch_texts.append(batch_titles[j])
                current_indices.append(i + j)

        except Exception as e:
            continue

    if len(current_batch_images) > 0:
        with torch.no_grad():
            inputs = processor(
                text=current_batch_texts,
                images=current_batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            
            outputs = model(**inputs)
            
            final_image_embeddings.append(outputs.image_embeds.cpu().numpy())
            final_text_embeddings.append(outputs.text_embeds.cpu().numpy())
            valid_indices.extend(current_indices)

            del inputs, outputs
            torch.cuda.empty_cache()  

if len(final_image_embeddings) > 0:
    combined_features = np.hstack((
        np.vstack(final_image_embeddings),
        np.vstack(final_text_embeddings)
    ))
    print(f"Final Shape: {combined_features.shape}")
else:
    print("ERROR")  

# %%
print(len(valid_indices))
print(len(final_image_embeddings))
print(len(final_text_embeddings))

# %%
print(combined_features.shape)

# %%
np.save('embeddings', combined_features)
print("saved")

# %%
# CREATING TWO SEPARATE SUB-DATASETS 
# One has data of the image containing posts and the other one has the data of the rest of the posts.

# Creating the Image Dataset 
df_images_only = df.iloc[valid_indices].reset_index(drop=True)

# Creating the Text-Only Dataset
all_indices = set(df.index)
image_indices = set(valid_indices)
text_only_indices = list(all_indices - image_indices)

df_text_only = df.iloc[text_only_indices].reset_index(drop=True)

# %%
df.shape

# %%
df_images_only.shape

# %%
df_text_only.shape

# %%
df.head()

# %%
df_images_only.head()

# %%
df_text_only.head()

# %%
df.describe()

# %%
df_images_only.describe()

# %%
df_text_only.describe()

# %%
df.isnull().sum()

# %%
df_images_only.isnull().sum()

# %%
df_text_only.isnull().sum()

# %%
# EDA 2 (Top 20 subreddits with maximum percentage of image containing posts)


df['has_image'] = df.index.isin(valid_indices)
subreddit_stats = df.groupby('subreddit')['has_image'].mean() * 100
subreddit_stats = subreddit_stats.sort_values(ascending=False)
top_20 = subreddit_stats.head(20)

plt.figure(figsize=(10, 8))  
bar_plot = sns.barplot(x=top_20.values, y=top_20.index, palette='viridis')

plt.title('Top 20 most subreddits with most number of image containing posts', fontsize=16)
plt.xlabel('Percent with Images', fontsize=12)
plt.ylabel('Subreddit', fontsize=12)
plt.grid(axis='x')
plt.xlim(0, 115) 
plt.tight_layout()
plt.show()

# %%
# EDA 3 (20 subreddits with lowest percentage of image containing posts)

bottom_20 = subreddit_stats.sort_values(ascending=True).head(20)
plt.figure(figsize=(10, 8))
bar_plot = sns.barplot(x=bottom_20.values, y=bottom_20.index, palette='magma') 

plt.title('Top 20 subreddits with least image containing posts', fontsize=16)
plt.xlabel('Percent with Images (%)', fontsize=12)
plt.ylabel('Subreddit', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.xlim(0, max(bottom_20.values) + 15) 
plt.tight_layout()
plt.show()


# %%
# EDA 4 (Scatter plot of num_comments VS score)

sns.scatterplot(
    x = 'score',
    y = 'num_comments',
    data = df,
    hue = 'has_image'
)
plt.show()


# %%
# EDA 5 (Log distribution of score and num_comments)


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1 - Score 
sns.histplot(df['score'], bins=50, ax=axes[0], color='blue')
axes[0].set_yscale('log') 
axes[0].set_title('Log distribution of score')
axes[0].set_xlabel('Score')

# Plot 2 - Comments 
sns.histplot(df['num_comments'], bins=50, ax=axes[1], color='green')
axes[1].set_yscale('log') 
axes[1].set_title('Log distribution of num_comments')
axes[1].set_xlabel('Number of Comments')
plt.tight_layout()
plt.show()




# FEATURE ENGINEERING

# %%
# Filling the empty body text rows
df['body_text'] = df['body_text'].fillna("") # for safety

# 1. Title Length (Character Count)
df['title_len'] = df['title'].str.len()

# 2. Body Length (Character Count)
df['body_len'] = df['body_text'].str.len()

# 3. Word Count (Token Count)
df['word_count'] = df['title'].apply(lambda x: len(str(x).split()))


# %%

# 4. Polarity (The VibeCheck)
# Range is -1(Negative emotions) to +1(Positive emotions)
df['sentiment_polarity'] = df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Subjectivity (fact vs opinion)
# Range: 0(Fact) to 1(Opinion)
df['sentiment_subjectivity'] = df['title'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# %%

# 5. Is the post a question 
question_starters = ('who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does')

df['is_question'] = df['title'].str.lower().apply(
    lambda x: 1 if str(x).strip().endswith('?') or str(x).startswith(question_starters) else 0
)

# %%

# 6. Checking if there are a lot of CAPS letters in the post
# High values of this ratio suggest either urgency or spam posts.
df['title_caps_ratio'] = df['title'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
)

# 7. Counting exclamation marks in the post
# High number of exclamation marks usually suggest high intensity emotions
df['exclamation_count'] = df['title'].apply(lambda x: str(x).count('!'))

# %%

# 8. Ratio between body and title
df['body_title_ratio'] = (df['body_len'] + 1) / (df['title_len'] + 1)

# %%
# 9. One hot encoding of subreddits
df_encoded = pd.get_dummies(df, columns=['subreddit'], prefix='sub', drop_first=True, dtype=int)

# %%
# 10 & 11. Average word length of title and body text

def avg_word_length(text):
    avg=[]
    for i in text:
        words = i.split()
        if len(words)>0:
            avg.append(sum(len(w) for w in words) / len(words)+1)
        else:
            avg.append(0)    
    return avg


df['avg_word_length_title'] = avg_word_length(df['title'])
df['avg_word_length_body'] = avg_word_length(df['body_text'])

# %%
# 12. Is post personal??

def personal_post(text):
    first_person_words = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    c=0
    for i in text:
        words = i.split()
        num_words = len(words)
        if words in first_person_words:
            c+=1

    return c / num_words

df['is_post_personal'] = personal_post(df['title'] + "" + df['body_text'])
# %%
# GENERATING TEXT EMBEDDINGS


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

text_vectors = model.encode((df['title'] + " " + df['body_text']).tolist(), show_progress_bar=True)

df['text_embedding'] = list(text_vectors)

# %%
text_vectors.shape


# %%
# CREATING COMBINED DATSET OF IMAGE + TEXT EMBEDDINGS


X = np.zeros((5223, 512 + 384)) 
# 5223 - Number of rows in the dataset
# 512 - Dimension of the feature vector generated by CLIP
# 384 - Dimension of the feature vector generated by MiniLM

X[:, 512:] = text_vectors

image_vectors = np.vstack(final_image_embeddings)
X[valid_indices, :512] = image_vectors

target_var = ['score', 'num_comments']
y = df[target_var].values

# %%

X_image = X[valid_indices]
y_image = y[valid_indices]


# %%
X_image.shape

# %%
text_indices = ~df.index.isin(valid_indices)

X_text = X[text_indices]
y_text = y[text_indices]

# %%
X_text.shape

# %%
# Adding the additional features to the dataset
new_features_columns = [
    'title_len', 'body_len', 'word_count', 
    'sentiment_polarity', 'sentiment_subjectivity', 
    'is_question', 'title_caps_ratio', 
    'exclamation_count', 'body_title_ratio',
    'avg_word_length_title', 'avg_word_length_body',
    'is_post_personal'
]

new_features = df[new_features_columns]

new_features_images = new_features.iloc[valid_indices].values
new_features_text = new_features.iloc[text_indices].values


subreddit_columns = [c for c in df_encoded.columns if c.startswith('sub_')]
subreddit_features = df_encoded[subreddit_columns].values

subreddit_features_images = subreddit_features[valid_indices]
subreddit_features_text = subreddit_features[text_indices]



# %%
new_features.shape
# %%
subreddit_features_images.shape
# %%
subreddit_features_text.shape
# %%
new_features_images.shape
# %%
new_features_text.shape



# %%
X_image_final = np.hstack(
    [X_image, new_features_images, subreddit_features_images]
)

X_text_final = np.hstack(
    [X_text, new_features_text, subreddit_features_text]
)


# %%
X_image_final.shape
# %%
X_text_final.shape

# %%
# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X_train_image, X_test_image, y_train_image, y_test_image = train_test_split(X_image_final, y_image, test_size=0.2, random_state=9)
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text_final, y_text, test_size=0.2, random_state=9)


# %%
# GRADIENT BOOSTING MODEL GENERATION

# Importing Libraries
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import numpy as np

# Creating model 1
def build_fast_stack():
    base_models = [
        ('hgb', HistGradientBoostingRegressor(
            max_iter=3000, 
            learning_rate=0.01, 
            l2_regularization=1.5
        )),
            
        ('xgb', XGBRegressor(
            tree_method='hist', 
            n_estimators=2000, 
            learning_rate=0.01
        ))
    ]

    ensemble = StackingRegressor(
        estimators=base_models,
        final_estimator=RidgeCV(),
        cv=3,
        n_jobs=-1
    )

    return TransformedTargetRegressor(
        regressor=ensemble,
        func=np.log1p,
        inverse_func=np.expm1
    )


# Creating model 2 (did not use this model as it was not giving that good results)
def build_ultimate_model():
    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=31,
        importance_type='gain',
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbode = 100
    )
    
    hgb_model = HistGradientBoostingRegressor(
        max_iter=3000,
        learning_rate=0.01,
        max_depth=12,
        l2_regularization=2.0,
        verbose = 1
    )
    
    # Combine them
    ensemble = VotingRegressor(estimators=[
        ('lgb', lgb_model),
        ('hgb', hgb_model)
    ])
    
    return TransformedTargetRegressor(
        regressor=ensemble,
        func=np.log1p,
        inverse_func=np.expm1
    )

# %%
# Training model with image dataset
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

model_image = MultiOutputRegressor(build_fast_stack())
model_image.fit(X_train_image, y_train_image)

y_pred_image = model_image.predict(X_test_image)

scores = r2_score(y_test_image, y_pred_image, multioutput='raw_values')

print(f"R2 (score): {scores[0]}")
print(f"R2 (num_coments): {scores[1]}")

# %%
# Training model with text dataset
model_text = MultiOutputRegressor(build_fast_stack())
model_text.fit(X_train_text, y_train_text)

y_pred_text = model_text.predict(X_test_text)

scores = r2_score(y_test_text, y_pred_text, multioutput='raw_values')

print(f"R2 (score): {scores[0]}")
print(f"R2 (num_coments): {scores[1]}")

# %%
# Predicting the overall R2 score

y_pred = np.vstack((y_pred_image, y_pred_text))
y_actual = np.vstack((y_test_image, y_test_text))

overall_score = r2_score(y_actual, y_pred, multioutput = 'raw_values')

print(f"R2 (score): {overall_score[0]:.4f}")
print(f"R2 (num_coments): {overall_score[1]:.4f}")

