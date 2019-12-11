# -*- coding: utf-8 -*-
import re


def preprocess_text(text):
    text = re.sub('[\(\[].*?[\)\]]', '', text)  # remove brackets
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    text = text.strip().strip('\"').strip()  # strip ""s and possible leftover whitespaces

    # Wiki summaries as sometimes structured like this:
    #
    # Some actual episode summary bla bla...
    # \n\n
    # Some "fun" fact about the directory or the actors or some note from the writer
    #
    # Obviously we want to get rid of the part after the \n-s
    text = text.split("\n")[0]

    return text


class EpisodeSummaryGeneratorPipeline(object):

    def process_item(self, item, spider):
        if item['source_url']:
            item['source_url'] = item['source_url'][0] 
        
        if item['episode_title']:
            item['episode_title'] = preprocess_text(item['episode_title'][0])

        if item['episode_summary']:
            item['episode_summary'] = preprocess_text(item['episode_summary'][0])

        if item['tv_show_title']:
            item['tv_show_title'] = preprocess_text(item['tv_show_title'][0])

        return item
