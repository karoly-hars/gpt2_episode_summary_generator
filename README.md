# Generating TV show episode summaries with GPT-2

With the code in this repo, it is possible to scrape IMDb and Wikipedia to acquire a large number of episode 
summaries for a TV show, and to use the data to train a GPT-2 model to generate similar summaries. Due to the lack of 
large-scale datasets, the generated summaries are usually far from perfect, but they are grammatically correct thanks to
the pre-training of the GPT-2 networks. Also, as expected, they usually quite funny :grin:.


## Requirements
To code was written/tested using python3 and the following packages:
- torch 1.3.1
- Beautifulsoup 4.8.0
- pytorch_transformers 1.2.0
- Scrapy 1.8.0

(plus their prerequisites)


## How-To
Clone the repo and install the required packages.


### Running the Scrapy spiders/crawlers

##### IMDb spider
Running the IMDb spider is fairly simple. It only takes a set of keywords as input params: 

Examples:

- You can download all the available episode data from all the "Star Trek"
TV shows into a JSON file by running ```python3 run_imdb_spider.py --title_keywords star trek```.

- Similarly, running ```python3 run_imdb_spider.py --title_keywords walker texas ranger``` downloads the episode data
for the TV show "Walker, Texas Ranger".

##### Wikipedia spider
The Wiki spider is slightly more complicated than the previous one. The user have to provide a starting page 
for the recursive search in Wikipedia, plus a string that will be used to filter out URLs 
(URLs that do not contain the string will be skipped) and a list of keywords,
that will be used to filter out more pages based on the title of the Wikipedia article. 

Examples:

- Run ```python3 run_wiki_spider.py --start_url https://en.wikipedia.org/wiki/Star_Trek --title_keywords star trek --url_substring Star_Trek```
to parse all the episode summaries from all "Star Trek" TV shows to a JSON file.

- Run ```python3 run_wiki_spider.py --start_url https://en.wikipedia.org/wiki/Walker,_Texas_Ranger --title_keywords walker texas ranger --url_substring Walker,_Texas_Ranger```
to do the same for "Walker, Texas Ranger".

---

In general, it is probably a good idea to choose TV shows or franchises with at least 100-150 episodes,
otherwise we might end up with subpar results (see the "Game of Thrones" example bellow).

After the spiders are finished, take a quick look into the output JSON files and verify that they contain the right data.

For additional information, check ```python3 run_imdb_spider -h``` and ```python3 run_wiki_spider -h```.

---

##### Pre-scapred episode data
I ran the spiders for some TV shows to provide an opportunity for users to train a network without running the spiders first.
The data can be found in ```./scraped_data```.


### Training the network
Call ```python3 train.py``` to train a GPT-2 network. The script splits the data into a training 
and validation subset. During the training, there is a checkpoint at every X step. At these checkpoints, the loss on 
the validation subset is calculated and a few samples are generated for the user to further monitor the progress.
The best model from the training is saved during the process.

The default params were selected based on what worked best for Star Trek episodes, so it might be a good idea to 
play around with the hyper-parameters a little bit if you are running the training on a different dataset, 
although, in my experience, the results are satisfactory even when training on a different set of data.

I ran my experiments on a GPU with ~16 GB memory. If you do not have the same resources, you will probably need to 
decrease the batch size or select a smaller GPT-2 model.

Multi-GPU training is currently not implemented.

For more information, check ```python3 train.py -h```.


### Generating episode summaries from scratch
After you have a trained model, you can generate episodes with
```python3 generate.py --num_samples <NUMBER_OF_SAMPLES_TO_GENERATE> --random_seed <SEED>```.

If you want to play around with other parameters of the generation process, check
```python3 generate.py -h```.

If you changed the GPT-2 version from the default ```gpt2-medium``` in the training, you will also have to changed 
it for the generation.


### Example outputs


#### [Star Trek](https://en.wikipedia.org/wiki/Star_Trek)
```
When Spock returns from a secret mission, he is captured by the enemy.
---
Sisko, Kira, Quark and Dax get trapped in a wormhole that leads to the Gamma Quadrant, and must use an ionic cloaking device to save themselves.
---
The Enterprise assists a small band of rebels from a planet in need of an emissary.
---
A damaged alien ship is found on a distant planet. The Enterprise crew is left with the difficult choice of trying to restore its power or destroy it to make way for a new ship.
---
The Enterprise encounters a ship of the Xindi, a race of advanced lifeformoids.
---
In an attempt to help a Klingon officer who was captured by the Federation, Chakotay, Kim, and Neelix learn of the Klingon commander's predicament, they are arrested in an effort to obtain a confession and return him to the Klingon Empire.
---
After Voyager encounters an unusual energy reading, Captain Janeway begins to suspect a traitor aboard the ship. Her suspicions become confirmed when she discovers several of the ship's computer systems are connected to different parts throughout the ship.
---
The Enterprise is ordered to a war zone in the Gamma Quadrant and Commander Tucker is brought up against resistance fighters of a rival faction. In an attempt to gain support, Tucker tells an impromptu story of his son's battle in World War II, in which his unit was ordered to a secret outpost near the German border. He is then captured by the resistance and is tortured for information.

```

#### [Walker, Texas Ranger](https://en.wikipedia.org/wiki/Walker,_Texas_Ranger)
```

```

#### [Charmed](https://en.wikipedia.org/wiki/Charmed)
```

```

#### [RuPaul's Drag Race](https://en.wikipedia.org/wiki/RuPaul's_Drag_Race)
```

```

#### [Game of Thrones](https://en.wikipedia.org/wiki/Game_of_Thrones)
```

```


## Acknowledgments
Some functions and code snippets were copied from https://github.com/huggingface/transformers
 