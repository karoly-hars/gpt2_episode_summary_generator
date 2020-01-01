import scrapy
from bs4 import BeautifulSoup
from utils.text_cleansing import clean_ep_data


class ImdbEpisodeSummarySpider(scrapy.Spider):

    name = 'imdb_episode_summary_spider'
    custom_settings = {'ROBOTSTXT_OBEY': False}

    def __init__(self, title_keywords='', *args, **kwargs):
        super(ImdbEpisodeSummarySpider, self).__init__(*args, **kwargs)

        self.allowed_domains = ['www.imdb.com']

        self.title_keywords = [word.lower() for word in title_keywords]
        start_search_url_substring = '+'.join(self.title_keywords)
        self.start_urls = ['https://www.imdb.com/find?q={}&s=tt&ref_=fn_al_tt_mr'.format(start_search_url_substring)]

    def parse(self, response):
        """Parse and yield all relevant episode data from an IMDb page."""
        tv_show_links = response.xpath(
            '//table[@class="findList"]/tr[contains(@class, "findResult ")]/td[@class="result_text"]/a/@href'
        ).extract()

        tv_show_titles = response.xpath(
            '//table[@class="findList"]/tr[contains(@class, "findResult ")]/td[@class="result_text"]'
        ).extract()

        for tv_show_title, tv_show_link in zip(tv_show_titles, tv_show_links):
            tv_show_title = BeautifulSoup(tv_show_title, 'lxml').get_text().strip()

            # go into the tv show pages. skip unrelated shows (not containing our keywords), and individual episodes
            if all([word.lower() in tv_show_title.lower() for word in self.title_keywords]) and \
                    '(tv series)' in tv_show_title.lower() and '(tv episode)' not in tv_show_title.lower():

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

                    episode_list_urls = [response.urljoin(e) for e in episode_list_urls if 'season' in e]

                    for url in episode_list_urls:
                        yield scrapy.Request(url, callback=self.parse_episode_list)

    def parse_episode_list(self, response):
        """Look up the episode urls from an episode list page."""
        episode_list = response.xpath('//div[@class="list detail eplist"]')
        episode_page_urls = episode_list.xpath('//div[@class="info"]/strong/a/@href').extract()
        episode_page_urls = ['https://www.imdb.com{}plotsummary'.format(e) for e in episode_page_urls]

        for url in episode_page_urls:
            yield scrapy.Request(url, callback=self.parse_plot_summary_page)

        # look up the page for the previous/next season
        prev = response.xpath('//div/a[@id="load_previous_episodes"]/@href').extract_first()
        next = response.xpath('//div/a[@id="load_next_episodes"]/@href').extract_first()

        base_url = response.url
        base_url = base_url.split("episodes?")[0]

        if prev:
            prev_url = "{}episodes{}".format(base_url, prev)
            yield scrapy.Request(prev_url, callback=self.parse_episode_list)
        if next:
            next_url = "{}episodes{}".format(base_url, next)
            yield scrapy.Request(next_url, callback=self.parse_episode_list)

    def parse_plot_summary_page(self, response):
        """Create and load the episode summary items."""
        show_title = response.xpath('//div[@class="subpage_title_block"]//h4/a/text()').extract_first().strip()
        ep_title = response.xpath('//div[@class="subpage_title_block"]//h3/a/text()').extract_first().strip()
        summaries = response.xpath('//ul[@id="plot-summaries-content"]/li[@class="ipl-zebra-list__item"]/p').extract()

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
