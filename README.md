# TV Show Episode Summary Generator

With the code in this repo, it is possible to scrape IMDb and Wikipedia 
to acquire a large number of episode summaries for a TV show, 
and to use the data to train a GPT-2 model to generate similar summaries.


## Requirements
To code was written/tested using python3 and the following packages:
- torch 1.3.1
- Beautifulsoup 4.8.0
- pytorch_transformers 1.2.0
- Scrapy 1.8.0

and their prerequisites.


## How-To
Clone the repo and install the required packages.


### Running the Scrapy spiders/crawlers


##### Running the IMDb spider
Running the IMDb spider is fairly simple. It takes two params as input: 
the path to the output JSON file, and the keywords for the title of the TV show.

Examples:

- Run ```python3 run_imdb_spider.py --title_keywords star trek``` to download the necessary
 data from all the [Star Trek](https://en.wikipedia.org/wiki/Star_Trek)
TV shows from IMDb into a JSON file.

- Run ```python3 run_imdb_spider.py --title_keywords walker texas ranger``` to 
get the episode data for the show [Walker, Texas Ranger](https://en.wikipedia.org/wiki/Walker,_Texas_Ranger).


##### Running the Wikipedia spider
The Wiki spider is slightly more complicated than the previous one.
The user have to provide a starting page for the recursive search in Wikipedia,
plus a string that will be used to filter out URLs 
(URL that do not contain the substring will not be skipped) and a list of keywords,
that will be used to filter out more pages based on the title of the Wikipedia article. 

Examples:

- Run ```python3 run_wiki_spider.py --start_url https://en.wikipedia.org/wiki/Star_Trek --title_keywords star trek --url_substring Star_Trek```
to parse all the episode summaries from all [Star Trek](https://en.wikipedia.org/wiki/Star_Trek) TV shows on Wikipedia to a JSON file.

- Run ```python3 run_wiki_spider.py --start_url https://en.wikipedia.org/wiki/Walker,_Texas_Ranger --title_keywords walker texas ranger --url_substring Walker,_Texas_Ranger```
to do the same for [Walker, Texas Ranger](https://en.wikipedia.org/wiki/Walker,_Texas_Ranger)


In general, it is probably a good idea to choose TV shows or franchises with at least 150-200 episodes,
otherwise we might end up with subpar results.

After the spiders are finished, it is a good idea to look into the output JSON files and verify that they contain the right data.

For additional information, check ```python3 run_imdb_spider -h``` and ```python3 run_wiki_spider -h```.


### Training the network
Call ```python3 train.py``` to train a GPT-2 network and save the weights.
The script splits the data into a training and validation subset. 
During the training, there is a checkpoint at every X step. At these checkpoints,
the loss on the validation subset is calculated and a few samples are generated 
for the user to further monitor the progress.
The best model from the training is saved after the process is finished.

The default params were selected based on what worked best for Star Trek episodes,
so it might be a good idea to play around with the hyper-parameters a little bit
if you are running the training on a different dataset. Although, in my experience, 
the results are satisfactory even when training on a different set of data.

I ran my experiments on a GPU with ~16 GB memory. If you do not have the same resources,
you will probably need to decrease the batch size or select a smaller GPT-2 model.

For more information, check ```python3 train.py -h```.

### Generating episode summaries from scratch
After you have a trained model, you can generate episodes with
```generate.py --num_samples <NUMBER_OF_SAMPLES_TO_GENERATE> --random_seed <SEED>```.

If you want to play around with other parameters of the generation process, check
```python3 generate.py -h```.

Example output for Star Trek:


Example output for Walker, Texas Ranger:
