import argparse
from scrapy.crawler import CrawlerProcess
from spiders.imdb_episode_summary_spider import ImdbEpisodeSummarySpider


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IMDb episode summary spider.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t', '--title_keywords', nargs='*', required=True,
                        help='Keywords from the TV show title. '
                             'Ideally they should be lowercase and whitspace separated.'
                             'Examples: \"star trek\" or \"rick and morty\"')
    parser.add_argument('-o', '--output_path', type=str, required=False, default='imdb_episode_summaries.json',
                        help='Path to the output JSON file. If the file already exists, it will be overwritten.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()

    # overwrite output. not too elegant, but there is no better way to do it at the moment.
    with open(args.output_path, "w") as f:
        pass

    # run spider
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': args.output_path,
    })
    process.crawl(ImdbEpisodeSummarySpider, title_keywords=args.title_keywords)
    process.start()
