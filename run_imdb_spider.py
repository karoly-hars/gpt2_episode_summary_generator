import os
import time
import argparse
import gzip
import shutil
import csv
from scrapy.crawler import CrawlerProcess
from spiders.imdb_episode_summary_spider import ImdbEpisodeSummarySpider


def download_and_uncompress_imdb_data(imdb_data_path):
    """Download and uncompress the basic title data from imdb.com/interfaces/."""
    imdb_gz_path = os.path.join(imdb_data_path, 'title.basics.tsv.gz')
    imdb_tsv_path = imdb_gz_path[:-3]

    # if the file does not exists or it is older than 1 day, download the update version
    if not os.path.exists(imdb_tsv_path) or time.time() - os.path.getmtime(imdb_tsv_path) > 86400:
        # download + uncompress
        os.system('wget https://datasets.imdbws.com/title.basics.tsv.gz -O {}'.format(imdb_gz_path))
        with gzip.open(imdb_gz_path, 'rb') as f_in:
            with open(imdb_tsv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return imdb_tsv_path


def get_start_urls(search_keywords, imdb_tsv_path):
    """Run a quick search to filter out possible start URLs for the spider."""
    # run some filtering on the title keywords
    search_keywords = [w.lower().strip() for w in search_keywords if len(w)]
    assert(len(search_keywords) > 0)

    # get start urls from the csv file
    start_urls = []
    with open(imdb_tsv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            title_type = row[1].lower()

            # extract tv shows only
            if title_type == 'tvseries':
                title = row[2].lower().split()

                # if we have multiple search keywords, check for titles containing all of them
                if len(search_keywords) > 1:
                    if all([any([search_kw in title_word for title_word in title]) for search_kw in search_keywords]):
                        start_urls.append('https://www.imdb.com/title/{}/'.format(row[0]))

                # if there is just 1 keyword, look for an exact match
                elif len(search_keywords) == 1:
                    if search_keywords == title:
                        start_urls.append('https://www.imdb.com/title/{}/'.format(row[0]))

    return start_urls


def run_imdb_spider(args):
    """Define and start process for IMDb scraping."""
    # download imdb data
    imdb_tsv_path = download_and_uncompress_imdb_data(args.imdb_data_path)

    # get start urls
    print('Preparing spider...')
    start_urls = get_start_urls(args.search_keywords, imdb_tsv_path)

    # if the search was unsuccessful or we have too many matches, display a message and return
    if not len(start_urls):
        print('No title matches were found in the IMDb dataset for keywords {}. Please refine search!'.format(
            args.search_keywords
        ))
        return
    elif len(start_urls) > 99:
        print('Too many ({}) title matches were found for keywords {}. Please refine search!'.format(
            len(start_urls), args.search_keywords
        ))
        return

    # overwrite output. not too elegant, but there is no better way to do it at the moment.
    with open(args.output_path, 'w') as f:
        pass

    # run spider
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': args.output_path,
        'ROBOTSTXT_OBEY': True
    })
    process.crawl(ImdbEpisodeSummarySpider, start_urls=start_urls)
    process.start()


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='IMDb episode summary spider.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--search_keywords', nargs='+', required=True,
                        help='Search keywords used for filtering TV shows based on their title. '
                             'The provided words should be whitespace separated. '
                             'Examples: "star trek" or "rick and morty"')
    parser.add_argument('-d', '--imdb_data_path', type=str, required=False, default='.',
                        help='Download and extraction path for the IMDb data subset used for URL extraction.')
    parser.add_argument('-o', '--output_path', type=str, required=False, default='imdb_episode_summaries.json',
                        help='Path to the output JSON file. If the file already exists, it will be overwritten.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    run_imdb_spider(args)
