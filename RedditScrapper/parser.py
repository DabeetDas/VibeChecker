import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reddit scraper for engagement dataset"
    )

    parser.add_argument(
        "--subreddit",
        type=str,
        required=True,
        help="Subreddit name (e.g. IITK)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        default=10,
        help="Number of posts to scrape"
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        default="reddit_posts.csv",
        help="Output CSV filename"
    )

    return parser.parse_args()