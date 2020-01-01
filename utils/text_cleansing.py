import re


def preprocess_title(text):
    """Pre-process an episode title string by removing unnecessary quotations, brackets, and whitespaces."""
    text = re.sub('[\(\[].*?[\)\]]', '', text)  # remove brackets
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    text = text.strip().strip('\"').strip()  # strip ""s and possible leftover whitespaces

    return text


def preprocess_summary(text):
    """Pre-process an episode summary string by removing repeated whitespaces, bracketed text, and citations."""
    text = re.sub('[\(\[].*?[\)\]]', '', text)  # remove brackets
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    text = re.sub('\s+\.\s+', '. ',  text)  # removed whitespaces from before dots

    # We want to get rid of the part after the first '\n' for summaries with multiple paragraphs
    text = text.split('\n')[0]

    # make sure the last sentence ends with '.', '!', or '?', if there is a half finished sentence that is usually a
    # citation or reference on Wikipedia
    if not (text.endswith('.') or text.endswith('?') or text.endswith('!')):
        last_closing = max([text.rfind('.'), text.rfind('?'), text.rfind('!')])
        if last_closing > 0:
            text = text[:last_closing+1]

    if text.endswith(' .'):
        text = text[:-2]+'.'

    return text


def clean_ep_data(ep_data):
    """Clean the episode data by running some simple cleansing on the titles and the summary."""
    if ep_data['episode_title']:
        ep_data['episode_title'] = preprocess_title(ep_data['episode_title'])

    if ep_data['episode_summary']:
        ep_data['episode_summary'] = preprocess_summary(ep_data['episode_summary'])

    if ep_data['tv_show_title']:
        ep_data['tv_show_title'] = preprocess_title(ep_data['tv_show_title'])

    return ep_data
