import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description="IMDb episode summary parser.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=75)
    )
    parser.add_argument('-t', '--title_keywords', nargs='*',
                        help='Keywords from the TV show title. '
                             'Ideally they should be lowercase and whitspace separated.'
                             '\nExamples: \"star trek\" or \"rick and morty\"')
    parser.add_argument('-o', '--output_path', type=str, required=False, default='imdb_episode_summaries.json',
                        help='Path to the output JSON file.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()

    call = 'scrapy crawl imdb_episode_summary_spider -a title_keywords="{}" -o {}'.format(' '.join(args.title_keywords),
                                                                                          args.output_path)
    os.system(call)
