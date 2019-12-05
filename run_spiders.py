import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_url', type=str, help='start URL for the spider.')
    parser.add_argument('-a', '--allow', type=str,
                        help='Links must include this substring otherwise the spider will not enter the URL.')
    parser.add_argument('-d', '--deny', type=str,
                        help='The spider will not enter links which include any of these substrings.')
    parser.add_argument('-t', '--title_keywords', nargs='*',
                        help='The title of the Wikipedia page must include these keywords, '
                             'otherwise the spider will not extract anything from the page.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    call = 'scrapy crawl wiki_episode_table_spider -a start_url=\"{}\" -a allow=\"{}\" ' \
           '-a deny=\"{}\" -a title_keywords=\"{}\" -o episode_summaries.json'.format(args.start_url,
                                                                                      args.allow,
                                                                                      args.deny,
                                                                                      ' '.join(args.title_keywords))
    os.system(call)
