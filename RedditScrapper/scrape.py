from helper import get_posts
from helper import get_comments_score_permalink
from helper import get_body_text_image
import pandas as pd
import time
from tqdm import tqdm
from parser import parse_args
import math

def main():
    args = parse_args()
    rows = []

    headers = {
        "User-Agent": "EngagementResearchBot/2.0 (academic)"
    }

    subreddit = args.subreddit
    max_pages = math.ceil(args.limit/25)
    posts = get_posts(headers,subreddit,max_pages)

    for post in tqdm(posts[:args.limit]):
        comments,score,permalink,title = get_comments_score_permalink(post)[0],get_comments_score_permalink(post)[1],get_comments_score_permalink(post)[2],get_comments_score_permalink(post)[3]
        time.sleep(5)
        body_text,image_url = get_body_text_image(permalink,headers)[0],get_body_text_image(permalink,headers)[1]
        
        rows.append({
            "subreddit": subreddit,
            "permalink": permalink,
            "score": score,
            "num_comments": comments,
            "title": title,
            "body_text": body_text,
            "image_url": image_url
        })

        time.sleep(2)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Saved dataset with", len(df), "rows")

if __name__ == "__main__":
    main()
