# -*- coding: utf-8 -*-
import re


def preprocess_text(text):
    text = re.sub('[\(\[].*?[\)\]]', '', text)  # remove brackets
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    text = text.strip()  # strip
    text = text.strip('\"')  # strip ""s
    return text


class EpisodeSummaryGeneratorPipeline(object):

    def process_item(self, item, spider):
        if item['source_url']:
            item['source_url'] = item['source_url'][0] 
        
        if item['episode_title']:
            item['episode_title'] = preprocess_text(item['episode_title'][0])

        if item['episode_summary']:
            item['episode_summary'] = preprocess_text(item['episode_summary'][0])

        return item
