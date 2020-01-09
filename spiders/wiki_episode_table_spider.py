from bs4 import BeautifulSoup
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from utils.text_cleansing import clean_ep_data


class WikiEpisodeTableSpider(CrawlSpider):
    """Crawler for collecting episode summaries by crawling through Wikipedia and parsing season/episode tables."""

    name = 'wiki_episode_table_spider'

    def __init__(self, start_url, allow, title_keywords, *args, **kwargs):
        super(WikiEpisodeTableSpider, self).__init__(*args, **kwargs)

        self.start_urls = [start_url]
        self.to_allow = allow
        self.title_keywords = [word.lower() for word in title_keywords]
        self.unique_episode_summaries = set()

        # set Wiki specific stuff
        self.allowed_domains = ['en.wikipedia.org']
        self.to_deny = ['/Talk:', '/Wikipedia_talk:', '/Category:', '/Wikipedia:', '/Template:']

        # set rules
        self.rules = (Rule(LinkExtractor(allow=self.to_allow,
                                         deny=self.to_deny,
                                         allow_domains=self.allowed_domains),
                           callback='parse_wiki_page',
                           follow=True),)

        super(WikiEpisodeTableSpider, self)._compile_rules()

    def parse_wiki_page(self, response):
        """Parse and yield all relevant episode data from a Wikipedia page."""
        page_header = response.xpath('//title').extract_first()

        if page_header and all([keyword in page_header.lower() for keyword in self.title_keywords]):
            # parse episode tables
            ep_tables = response.xpath('//table[@class="wikitable plainrowheaders wikiepisodetable"]')
            ep_titles, ep_sums = self.parse_episode_tables(ep_tables)

            if not ep_sums:
                return

            # prettify data
            ep_sums = [BeautifulSoup(ep_sum, 'lxml').get_text().strip() for ep_sum in ep_sums]
            ep_titles = [BeautifulSoup(ep_title, 'lxml').get_text().strip() for ep_title in ep_titles]

            # parse unique items
            for ep_title, ep_sum in zip(ep_titles, ep_sums):
                if ep_sum not in self.unique_episode_summaries:
                    self.unique_episode_summaries.add(ep_sum)

                    ep_data = {
                        'source_url': response.url,
                        'episode_title': ep_title,
                        'episode_summary': ep_sum,
                        'tv_show_title': '',
                    }

                    yield clean_ep_data(ep_data)

    def parse_episode_tables(self, ep_tables):
        """Collect the data from a list of Wikipedia episode tables into a list of titles and a list of summaries."""
        ep_titles = []
        ep_sums = []

        for ep_table in ep_tables:
            ep_table_titles = ep_table.xpath('.//tbody/tr[@class="vevent"]/td[@class="summary"]').extract()
            ep_table_sums = ep_table.xpath('.//tbody/tr[@class="expand-child"]/td[@class="description"]').extract()

            # some episode tables only contain the episode titles + other information, but not the ep summaries
            # also, so episode tables have missing summaries for some (yet to be aired) episodes. SKIP these.
            if not ep_table_sums or not ep_table_titles:
                continue

            for i in range(len(ep_table_sums)):
                ep_sums.append(ep_table_sums[i])
                ep_titles.append(ep_table_titles[i])

        return ep_titles, ep_sums
