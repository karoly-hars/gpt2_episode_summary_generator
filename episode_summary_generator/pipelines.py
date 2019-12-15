# -*- coding: utf-8 -*-
import re


def preprocess_title(text):
    text = re.sub('[\(\[].*?[\)\]]', '', text)  # remove brackets
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    text = text.strip().strip('\"').strip()  # strip ""s and possible leftover whitespaces
    return text


def preprocess_summary(text):
    text = re.sub('[\(\[].*?[\)\]]', '', text)  # remove brackets
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    text = re.sub('\s+\.\s+', '. ',  text)  # removed whitespaces from before dots

    # wiki summaries as sometimes structured like this:
    # "Some actual episode summary bla bla...
    # \n\n
    # Some "fun" fact about the directory or the actors or some note from the writer."
    # Obviously we want to get rid of the part after the \n-s
    text = text.split("\n")[0]

    # make sure the last sentence ends with '.', '!', or '?', if there is a half finished sentence that is usually a
    # citation or reference on Wikipedia
    if not (text.endswith(".") or text.endswith("?") or text.endswith("!")):
        last_closing = max([text.rfind('.'), text.rfind('?'), text.rfind('!')])
        if last_closing > 0:
            text = text[:last_closing+1]

    if text.endswith(" ."):
        text = text[:-2]+"."

    return text


class EpisodeSummaryGeneratorPipeline(object):

    def process_item(self, item, spider):
        if item['source_url']:
            item['source_url'] = item['source_url'][0] 
        
        if item['episode_title']:
            item['episode_title'] = preprocess_title(item['episode_title'][0])

        if item['episode_summary']:
            item['episode_summary'] = preprocess_summary(item['episode_summary'][0])

        if item['tv_show_title']:
            item['tv_show_title'] = preprocess_title(item['tv_show_title'][0])

        return item
