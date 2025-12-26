from bs4 import BeautifulSoup
import requests
import time
import cv2
import numpy as np



def get_posts(headers: dict[str, str], subreddit: str, max_pages: int):
    """
    Gets top posts (all time) from old.reddit from all pages according to the page limit.
    """
    all_posts = []
    after = None
    count = 0

    for page in range(max_pages):
        if after:
            url = (
                f"https://old.reddit.com/r/{subreddit}/top/"
                f"?sort=top&t=all&count={count}&after={after}"
            )
        else:
            url = f"https://old.reddit.com/r/{subreddit}/top/?sort=top&t=all"

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            posts = soup.select(".thing")

            if not posts:
                break 

            all_posts.extend(posts)

            
            after = posts[-1].get("data-fullname")
            count += len(posts)

            time.sleep(2)  

        except Exception as e:
            print(f"[ERROR] {e}")
            break

    return all_posts

'''

def get_posts(headers:dict[str,str],subreddit:str):
    
    Gets posts from  "all time top posts" category and returns all the posts
    

    try:
        url = f"https://old.reddit.com/r/{subreddit}/top/?sort=top&t=all"
        html = requests.get(url, headers=headers).text
        soup = BeautifulSoup(html, "lxml")
        posts = soup.select(".thing")
    except Exception as e:
        print(f"[UNKNOWN ERROR] {e}")


    return posts
'''

def get_comments_score_permalink(post):
    """
    Returns the number of comments, score(upvotes-downvotes) and the permalink of the reddit post
    """

    thing = post
    title = thing.select_one("a.title").get_text(strip=True)
    score = int(thing["data-score"])
    num_comments = int(thing["data-comments-count"])
    permalink = "https://old.reddit.com" + thing["data-permalink"]

    return (num_comments,score,permalink,title)

def get_body_text_image(permalink,headers:dict[str,str]):
    body_text = None
    image_url = None
    image_urls = None

    try:      
        post_html = requests.get(permalink, headers=headers).text
        post_soup = BeautifulSoup(post_html, "lxml")
        expando = post_soup.select_one("div.expando")

        if expando:
            usertext = expando.select_one("div.usertext-body div.md")
            image_div = expando.select_one("div.media-preview-content")
            gallery_images = expando.select("div.gallery-tile-content img")

            if not usertext:
                body_text = None
            else:
                body_text = usertext.get_text("\n",strip=True)
            
            image_url = None
            if image_div:
                link = image_div.select_one("a")
                if link and link.has_attr("href"):
                    image_url = link["href"]
            
            if gallery_images:
                image_urls = [
                    img["src"]
                    for img in gallery_images
                    if img.has_attr("src")
                ]
        
        if image_urls:
            image_url = image_urls
    
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] {permalink}")

    except requests.exceptions.ConnectionError:
        print(f"[CONNECTION ERROR] {permalink}")

    except requests.exceptions.HTTPError as e:
        print(f"[HTTP ERROR] {e} | {permalink}")

    except Exception as e:
        print(f"[UNKNOWN ERROR] {e} | {permalink}")
    
    return (body_text,image_url)
