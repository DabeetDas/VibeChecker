# Reddit Engagement Scraper

This tool allows you to scrape top posts from a specific subreddit on **old.reddit.com**. It collects engagement metrics (score, comments), post content, and images, saving the data into a structured CSV file for analysis.

## Directory Structure

Ensure your project folder contains these three specific files:

1.  **`scrape.py`**: The main script you will run. It manages the flow of the program and saves the final CSV.
2.  **`parser.py`**: Handles the command-line arguments (settings) you provide, such as the subreddit name and post limit.
3.  **`helper.py`**: Contains the core logic to connect to Reddit, parse HTML, extract text/images, and handle network requests.


## Prerequisites & Installation

To run this code locally, you need **Python** installed on your computer. You also need to install the external libraries that the code relies on.

### 1. Install Python
If you haven't already, download and install Python from [python.org](https://www.python.org/).

### 2. Install Dependencies
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run the following command. This installs all necessary packages found in the script imports (pandas, requests, bs4, lxml, tqdm, opencv, numpy):
```bash
pip install pandas requests beautifulsoup4 lxml tqdm opencv-python numpy
```

## How to Use
You will run the scraper using your **terminal**. The script requires you to specify **three** pieces of information: the subreddit name, how many posts to get, and where to save the file.

### Basic Command Format
Navigate to the folder containing your files in the terminal, then run:
```bash
python scrape.py --subreddit [NAME] --limit [NUMBER] --out [FILENAME.csv]
```
**Example**
To scrape the top 50 posts from the IITK subreddit and save them to my_data.csv:
```bash 
python scrape.py --subreddit IITK --limit 50 --out my_data.csv
```

### Explanation of arguments
All three arguments below are required:
| Argument     | Description                                                                 | Example          |
|-------------|-----------------------------------------------------------------------------|------------------|
| `--subreddit` | The name of the subreddit you want to scrape (do not include `r/`).        | `IITK`           |
| `--limit`     | The total number of posts you want to collect.                              | `100`            |
| `--out`       | The name of the CSV file where data will be saved.                          | `reddit_data.csv` |

## What to do?
You will be assigned subreddits that you have to scrape using the code provided. Your tasks are as follows:
1. Read and understand the entire code which we are using for scraping reddit.  
2. Scrape the subreddit assigned to you and upload the csv file in the Week-3 folder with the name ```Name_Subrredit.csv```.  
3. Although the code is tested, you may edit it locally if you face any errors.


