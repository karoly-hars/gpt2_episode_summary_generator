import scrapy
from bs4 import BeautifulSoup
from utils.text_cleansing import clean_ep_data


class ImdbEpisodeSummarySpider(scrapy.Spider):
    """Spider for scraping the episode summaries of a TV show on IMDb."""

    name = 'imdb_episode_summary_spider'

    def __init__(self, start_urls, *args, **kwargs):
        super(ImdbEpisodeSummarySpider, self).__init__(*args, **kwargs)

        self.allowed_domains = ['www.imdb.com']
        self.start_urls = start_urls

    def parse(self, response):
        """Look up the episode list urls."""
        # get the rating count. if its > 500, this might be a real TV show, not some fan-made project
        # this does not work all the time, but in my experience it might be a good indicator
        rating_count = response.xpath('//*[@itemprop="ratingCount"]/text()').extract_first()

        if rating_count:
            rating_count = int(rating_count.replace(',', ''))

            if rating_count > 500:
                # look up episodes
                episode_list_urls = response.xpath('//*[@class="seasons-and-year-nav"]/div/a/@href').extract()
                episode_list_urls = [response.urljoin(e) for e in episode_list_urls if 'season' in e]

                for url in episode_list_urls:
                    yield scrapy.Request(url, callback=self.parse_episode_list)

    def parse_episode_list(self, response):
        """Look up the episode urls from an episode list page."""
        episode_list = response.xpath('//*[@class="list detail eplist"]')
        episode_page_urls = episode_list.xpath('//*[@class="info"]/strong/a/@href').extract()
        episode_page_urls = ['https://www.imdb.com{}'.format(e) for e in episode_page_urls]

        for url in episode_page_urls:
            yield scrapy.Request(url, callback=self.parse_episode_page)

        # look up the page for the previous/next season
        prev = response.xpath('//*[@id="load_previous_episodes"]/@href').extract_first()
        next = response.xpath('//*[@id="load_next_episodes"]/@href').extract_first()

        base_url = response.url
        base_url = base_url.split("episodes?")[0]

        if prev:
            prev_url = "{}episodes{}".format(base_url, prev)
            yield scrapy.Request(prev_url, callback=self.parse_episode_list)

        if next:
            next_url = "{}episodes{}".format(base_url, next)
            yield scrapy.Request(next_url, callback=self.parse_episode_list)

    def parse_episode_page(self, response):
        """Look up the link to the plot summary page from episode page and process it accordingly."""
        plot_summary_url = response.xpath('//*[text()="Plot Summary"]/@href').extract_first()
        plot_summary_url = 'https://www.imdb.com{}'.format(plot_summary_url)

        yield scrapy.Request(plot_summary_url, callback=self.parse_plot_summary_page)

    def parse_plot_summary_page(self, response):
        """Create and load the episode summary items."""
        show_title = response.xpath('//*[@class="subpage_title_block"]//h4/a/text()').extract_first().strip()
        ep_title = response.xpath('//*[@class="subpage_title_block"]//h3/a/text()').extract_first().strip()
        summaries = response.xpath('//*[@class="ipl-zebra-list__item" and contains(@id, "summary")]/p').extract()

        for ep_sum in summaries:
            ep_sum = BeautifulSoup(ep_sum, 'lxml').get_text().strip()

            if 'be the first to contribute' not in ep_sum.lower():
                ep_data = {
                    'source_url': response.url,
                    'episode_title': ep_title,
                    'episode_summary': ep_sum,
                    'tv_show_title': show_title,
                }

                yield clean_ep_data(ep_data)
