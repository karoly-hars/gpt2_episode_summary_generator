import os
import argparse


def get_arguments():
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

    # call spider
    call = 'scrapy runspider imdb_episode_summary_spider.py -a title_keywords="{}" -t json -o - > "{}"'
    os.system(call.format(' '.join(args.title_keywords), args.output_path))
