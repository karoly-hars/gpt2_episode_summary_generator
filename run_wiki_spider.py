import argparse
from scrapy.crawler import CrawlerProcess
from spiders.wiki_episode_table_spider import WikiEpisodeTableSpider


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Wikipedia episode summary spider.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--start_url', type=str, required=True,
                        help='start URL for the spider.'
                             'Should be: '
                             'https://en.wikipedia.org/wiki/<Show_Title_With_Underscores_And_Capitalized_Words>. '
                             'Example: https://en.wikipedia.org/wiki/Star_Trek')
    parser.add_argument('-u', '--url_substring', type=str, required=True,
                        help='Wikipedia urls must include this substring otherwise the spider will not enter the URL.'
                             'Ideally, it should be something like: '
                             '<Show_Title_With_Underscores_And_Capitalized_Words>. Example: "Star_Trek"')
    parser.add_argument('-t', '--title_keywords', nargs='*', required=True,
                        help='The title of the Wikipedia page must include these keywords, '
                             'otherwise the spider will not extract anything from the page. '
                             'Good practice: use the lowercase version of the words from the title of the show. '
                             'Example: star trek')
    parser.add_argument('-o', '--output_path', type=str, required=False, default='wiki_episode_summaries.json',
                        help='Path to the output JSON file. If the file already exists, it will be overwritten.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()

    # overwrite output
    with open(args.output_path, 'w') as f:
        pass

    # run spider
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': args.output_path
    })
    process.crawl(
        WikiEpisodeTableSpider, start_url=args.start_url, allow=args.url_substring, title_keywords=args.title_keywords
    )
    process.start()
