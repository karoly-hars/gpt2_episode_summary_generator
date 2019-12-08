# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.loader import ItemLoader
from episode_summary_generator.items import EpisodeSummaryGeneratorItem


class WikiEpisodeTableSpider(CrawlSpider):
    name = 'wiki_episode_table_spider'

    # set obey robots to True for Wikipedia
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
    }
    
    def __init__(self, start_url, allow, deny='', title_keywords='', *args, **kwargs):
        super(WikiEpisodeTableSpider, self).__init__(*args, **kwargs)

        self.allowed_domains = ['en.wikipedia.org']
        self.start_urls = [start_url]
        self.to_allow = allow
        self.title_keywords = [word.lower() for word in title_keywords.strip().split()]
        # this will be used to get rid of some nasty redirects by Wikipedia:
        self.to_deny = ["/Talk:", "/Wikipedia_talk:", "/Category:", "/Wikipedia:", "/Template:"] + \
                       [word for word in deny.strip().split()]

        self.num_episodes = 0
        self.unique_episode_summaries = set()

        # set rules
        self.rules = (Rule(LinkExtractor(allow=self.to_allow,
                                         deny=self.to_deny,
                                         allow_domains=self.allowed_domains),
                           callback='parse_item',
                           follow=True),)

        super(WikiEpisodeTableSpider, self)._compile_rules()

    def parse_item(self, response):
        page_header = response.xpath('//title').extract_first()

        if page_header and all([keyword in page_header.lower() for keyword in self.title_keywords]):       
            # parse episode tables
            ep_tables = response.xpath('//table[@class="wikitable plainrowheaders wikiepisodetable"]')
            ep_titles, ep_sums = self.parse_episode_tables(ep_tables)

            if not ep_sums:
                return

            # prettify data
            ep_sums = [BeautifulSoup(ep_sum, "lxml").get_text().strip() for ep_sum in ep_sums]
            ep_titles = [BeautifulSoup(ep_title, "lxml").get_text().strip() for ep_title in ep_titles]

            # parse unique items
            for ep_title, ep_sum in zip(ep_titles, ep_sums):
                if ep_sum not in self.unique_episode_summaries:
                    self.num_episodes += len(ep_sums)
                    self.unique_episode_summaries.add(ep_sum)

                    loader = ItemLoader(item=EpisodeSummaryGeneratorItem())
                    loader.add_value("source_url", response.url)
                    loader.add_value("episode_title", ep_title)
                    loader.add_value("episode_summary", ep_sum)
                    # in this case, the show title can be None, since it derivable from the url
                    loader.add_value("tv_show_title", "")

                    yield loader.load_item()

    def parse_episode_tables(self, ep_tables):
        ep_titles = []
        ep_sums = []

        for ep_table in ep_tables:
            ep_table_titles = ep_table.xpath('.//tbody/tr[@class="vevent"]/td[@class="summary"]').extract()
            ep_table_sums = ep_table.xpath('.//tbody/tr[@class="expand-child"]/td[@class="description"]').extract()

            # some episode tables only contain the episode titles + other information, but not the ep summaries
            # also, so episode tables have missing summaries for some (yet to be aired) episodes
            if not ep_table_sums or not ep_table_titles:
                continue

            for i in range(len(ep_table_sums)):
                ep_sums.append(ep_table_sums[i])
                ep_titles.append(ep_table_titles[i])

        return ep_titles, ep_sums