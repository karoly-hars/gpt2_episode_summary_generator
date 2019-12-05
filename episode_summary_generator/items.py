# -*- coding: utf-8 -*-
import scrapy


class EpisodeSummaryGeneratorItem(scrapy.Item):
    source_url = scrapy.Field()
    episode_title = scrapy.Field()
    episode_summary = scrapy.Field()
