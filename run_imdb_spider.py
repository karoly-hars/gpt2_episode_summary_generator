import os
import argparse
import gzip
import shutil
import csv
from scrapy.crawler import CrawlerProcess
from spiders.imdb_episode_summary_spider import ImdbEpisodeSummarySpider


def get_start_urls(title_keywords, imdb_data_path):
    """Download some title data from IMDb, and run a quick search to filter out possible start URLs for the spider."""
    # download
    imdb_gz_path = os.path.join(imdb_data_path, 'title.basics.tsv.gz')
    os.system('wget https://datasets.imdbws.com/title.basics.tsv.gz -O {}'.format(imdb_gz_path))

    # uncompress
    imdb_tsv_path = imdb_gz_path[:-3]
    with gzip.open(imdb_gz_path, 'rb') as f_in:
        with open(imdb_tsv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # get start urls from the csv file
    start_urls = []
    with open(imdb_tsv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            title_type = row[1].lower()

            # extract tv shows only
            if title_type == 'tvseries':
                title = row[2].lower()

                # if we have multiple title keywords, check for titles containing all of them
                if len(title_keywords) > 1:
                    if all([word in title for word in title_keywords]):
                        start_urls.append('https://www.imdb.com/title/{}/'.format(row[0]))

                # if there is just 1 keyword, look for an exact match
                elif len(title_keywords) == 1:
                    if title_keywords[0] == title:
                        start_urls.append('https://www.imdb.com/title/{}/'.format(row[0]))

    return start_urls


def run_imdb_spider(args):
    """Define and start process for IMDb scraping."""
    # run some filtering on the title keywords
    title_keywords = [w.lower().strip() for w in args.title_keywords if len(w)]
    assert(len(title_keywords) > 0)

    # download imdb data and get start urls
    start_urls = get_start_urls(title_keywords, args.imdb_data_path)

    # overwrite output. not too elegant, but there is no better way to do it at the moment.
    with open(args.output_path, 'w') as f:
        pass

    # run spider
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': args.output_path,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0'
    })
    process.crawl(ImdbEpisodeSummarySpider, start_urls=start_urls)
    process.start()


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='IMDb episode summary spider.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t', '--title_keywords', nargs='+', required=True,
                        help='Keywords from the TV show title. '
                             'Ideally they should be lowercase and whitespace separated.'
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
