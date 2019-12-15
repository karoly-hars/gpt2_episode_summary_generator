# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup
from scrapy.loader import ItemLoader
from episode_summary_generator.items import EpisodeSummaryGeneratorItem


class ImdbEpisodeSummarySpiderSpider(scrapy.Spider):
    name = 'imdb_episode_summary_spider'

    allowed_domains = ['www.imdb.com']

    def __init__(self, title_keywords='', *args, **kwargs):
        super(ImdbEpisodeSummarySpiderSpider, self).__init__(*args, **kwargs)

        self.title_keywords = [word.lower() for word in title_keywords.strip().split()]
        start_search_url_substring = '+'.join(self.title_keywords)
        self.start_urls = ['https://www.imdb.com/find?q={}&s=tt&ref_=fn_al_tt_mr'.format(start_search_url_substring)]

    def parse(self, response):
        tv_show_links = response.xpath(
            '//table[@class="findList"]/tr[contains(@class, "findResult ")]/td[@class="result_text"]/a/@href'
        ).extract()

        tv_show_titles = response.xpath(
            '//table[@class="findList"]/tr[contains(@class, "findResult ")]/td[@class="result_text"]'
        ).extract()

        for tv_show_title, tv_show_link in zip(tv_show_titles, tv_show_links):
            tv_show_title = BeautifulSoup(tv_show_title, "lxml").get_text().strip()

            # go into the tv show pages. skip unrelated shows (not containing our keywords), and individual episodes
            if all([word in tv_show_title.lower() for word in self.title_keywords]) and \
                    "(tv series)" in tv_show_title.lower() and "(tv episode)" not in tv_show_title.lower():

                next_page_url = response.urljoin(tv_show_link)
                yield scrapy.Request(url=next_page_url, callback=self.parse_tv_show_page)

    def parse_tv_show_page(self, response):
        """Look up the episode list urls."""
        ratings_wrapper = response.xpath('//div[@class="ratings_wrapper"]')

        if ratings_wrapper:
            # get the rating count. if its > 500, this might be a real TV show, not some fan-made project
            # this does not work all the time, but in my experience it might be a good indicator
            rating_count = ratings_wrapper.xpath('//span[@itemprop="ratingCount"]/text()').extract_first()
            if rating_count:
                rating_count = int(rating_count.replace(',', ''))
                if rating_count > 500:

                    # look up episodes
                    episode_list_urls = response.xpath(
                        '//div[@id="title-episode-widget"]/div[@class="seasons-and-year-nav"]/div/a/@href'
                    ).extract()

                    episode_list_urls = [response.urljoin(e) for e in episode_list_urls if "season" in e]

                    for url in episode_list_urls:
                        yield scrapy.Request(url, callback=self.parse_episode_list)

    def parse_episode_list(self, response):
        """Look up the episode urls."""
        episode_list = response.xpath('//div[@class="list detail eplist"]')
        episode_page_urls = episode_list.xpath('//div[@class="info"]/strong/a/@href').extract()
        episode_page_urls = ["https://www.imdb.com{}plotsummary".format(e) for e in episode_page_urls]

        for url in episode_page_urls:
            yield scrapy.Request(url, callback=self.parse_plot_summary_page)

    def parse_plot_summary_page(self, response):
        """Create and load the episode summary items."""
        show_title = response.xpath('//div[@class="subpage_title_block"]//h4/a/text()').extract_first().strip()
        ep_title = response.xpath('//div[@class="subpage_title_block"]//h3/a/text()').extract_first().strip()
        summaries = response.xpath('//ul[@id="plot-summaries-content"]/li[@class="ipl-zebra-list__item"]/p').extract()

        for ep_sum in summaries:
            ep_sum = BeautifulSoup(ep_sum, "lxml").get_text().strip()

            if "be the first to contribute" not in ep_sum.lower():
                loader = ItemLoader(item=EpisodeSummaryGeneratorItem())
                loader.add_value("source_url", response.url)
                loader.add_value("episode_title", ep_title)
                loader.add_value("episode_summary", ep_sum)
                loader.add_value("tv_show_title", show_title)
                yield loader.load_item()
