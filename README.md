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
otherwise we might end up with subpar results.

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
The Mirror Universe crew discovers a crashed Federation vessel that was used by Klingons, Borg and humans in a war to steal the ship's secrets.
---
Odo has been assigned to interrogate a former crewman who was captured in an ancient ritual.
---
An ion storm wipes out Voyager's transporters and sets the crew on the path to becoming a cyborg.
---
O'Brien is surprised when he gets a call from a woman who claims to be a Cardassian. She claims to have been kidnapped a year earlier by a Cardassian, whom she claims tortured her for years.
---
The Enterprise arrives at the home world of a young woman named Vina. She is an Earth native who has been raised in a Roman Catholic family since childhood and never once had any contact with a non-believer. She seems to have inherited her mother's sense of humor, and she enjoys her time with others.
---
A Bajoran woman comes aboard to claim her ancestor's body.
---
Sisko, Odo and Dax are on a holonovel about an evil Klingon named Kolzon. Kolzon is the most evil of the Klingons and he uses his psychic powers to torture people.
---
Dr. Crusher is captured by the Cardassians, and when the crew find out she is a terrorist infiltrating their society, they believe she was involved in espionage. She is found and tortured as a terrorist, and is eventually brought to the Cardassia Prime headquarters.
```

#### [Walker, Texas Ranger](https://en.wikipedia.org/wiki/Walker,_Texas_Ranger)
```
Walker and Trivette go undercover with a Mexican drug lord as he tries to get back at the Rangers with an assassination attempt.
---
Walker and Trivette are on their way to an old friend of Trivette's, and a Mexican Mafia leader. The leader of the gang was a drug dealer who tried to kill Walker when he was still in prison. But when Walker is released, the drug dealer is in prison again and the gang leader tells Trivette that they are not going through. He then kidnaps Trivette's friend and the friend is forced to help Trivette. When Walker and Trivette discover that the man who kidnapped the friend and the leader had ties to the Mexican Mafia, he sends them to kill him as well.
---
A man's son is killed by a group of men while they are camping. Walker tracks down the man who murdered his son. Meanwhile, Alex is at a party and is kidnapped by a group of guys. Alex is then held hostage by the group who threaten to kill her.
---
Walker and Trivette make the perilous trek across the desert to stop a group of Mexican outlaws who've kidnapped his son after he's rescued from being beaten by a rival gang. They also learn there's a deadly ring involving a high-ranking Mexican government agent's wife and two children whom the outlaws use to threaten their enemies.
---
Walker and Trivette take an old family friend from his life. The man reveals a secret he was afraid to speak out about before dying. A local politician who had a personal vendetta against Walker's family is also in the room. Walker and Trivette are able to convince him to go into exile.
---
Walker, who has been tracking down the mob boss behind the crime bosses, is in a car accident. He survives but his driver is killed. His partner Walker is taken prisoner to be killed. They get out alive but he is wounded in the head.
---
After a man who helped build a new jail, is killed in a shootout with guards in a failed attempt, Walker and Trivette attempt to bring to justice the man who ordered the attack.
---
Walker helps him escape from the police station. After they return from the hospital, Walker and Trivette are asked to attend a memorial for the woman she helped kill.
```

#### [Charmed](https://en.wikipedia.org/wiki/Charmed)
```
Phoebe Halliwell must stop a warlock who has been killing local witches in the tradition of the Evil Enchantress, whose wand she has kept hidden.
---
When Wyatt inadvertently brings the ghost of a serial killer to life, the sisters must bring him back to life or be destroyed.
---
A demon named Victor, employed by the sisters to confuse them into giving him the Leprechauns, steals their wands. Phoebe casts a spell that turns Paige, Piper and Leo into walking corpses, but ends up turning Leo into a ghost.
---
Piper and Leo plan a date, and Piper gets a premonition of Leo getting shot. Phoebe sees someone shooting and decides to get some information before Piper goes to the hospital.
---
Piper and Leo make their home for the first time, and Piper finds a room with a big TV in it. Piper notices an old painting by the name of "Dancing Queen" by Andy Warhol and thinks it is a good idea to have it done in the style of Warhol's painting "The Call of the Wild."
---
While on the hunt for the Charmed Ones, Cole gets a surprise from the Book of Shadow where he encounters an evil warlock from the future named Kyle Sheridan and a mysterious warlock from the future named Alec Williamson.
---
Phoebe and Paige discover a book that they think will help them deal with their Power of 3. After a spell is cast on Piper and Leo, the sisters discover that it is actually a Power of 4 that has been switched.
---
During Chinese New Year, the Halliwell sisters are attacked by a vicious demon known as a 'Banshee'. Piper is left in danger when her Chinese astral self is attacked and scratched, and Phoebe is left in despair when her Chinese astral self is killed. After learning about Chinese, Phoebe casts a spell to have a better understanding of Chinese and ends up getting her hands on a mysterious 'Manual of Chinese Charmed Exercises'.
```

#### [RuPaul's Drag Race](https://en.wikipedia.org/wiki/RuPaul's_Drag_Race)
```
The queens have a roast of the week with guest judges Cheryl Fernandez Valentino, RuPaul, and Miss Congeniality. On this episode, they get emotional talking about the moments that made them and the other queens stand out from the crowd.
---
The queens' runway looks are off-limits, as they must perform in an improv comedy act and create a parody of RuPaul's hit song "Queens Everywhere."
---
After the queens compete for their "biggest and best acting gig ever" in a RuPaul's Drag Race parody, RuPaul makes an appearance to give them some advice.
---
The queens discuss who they think should face the music and who might walk away with the crown. RuPaul reveals some shocking backstage knowledge.
---
In the mini challenge, the queens make puppets of two drag queens, portraying them, and drag queens from both sides of the political divide; the other queens are tasked to create their own puppets of each member of the pair.
---
This week's winner, Valentina is announced as the winner of the title of RuPaul's Next Drag Superstar.
---
The queens are tasked by RuPaul to create couture out of materials found at various flea markets, and transform them into something completely different: a "Sizzly Couture Party."
---
The queens work toward an epic mini-challenge: creating iconic runway looks, and acting in a RuPaul musical. Guest Judges: RuPaul and Gaby Hoffmann.
```

#### [Game of Thrones](https://en.wikipedia.org/wiki/Game_of_Thrones)
```
Tyrion's army is defeated at the Twins. Stannis is defeated at the Twins. Jon Snow takes Stannis prisoners. Daenerys Targaryen plans to conquer Westeros and sends her army to the North.
---
Arya is rescued by the survivors and she tells about the events at the Eyrie. Catelyn is forced to deal with the aftermath of the Great River Crossing incident, which leaves her with a scar that she has to hide from the Lannister family. Jon and the Night's Watch are attacked by Stannis' forces as he makes his escape, and he meets an unexpected visitor. Meanwhile, the Ironborn continue to fight against the Riverlands in the south.
---
Theon has been captured and killed by the Boltons but Sam has escaped. Tyrion advises Margaery about her sister and she agrees to travel to meet her. Jon decides to go with Rickard to the Wall. Daenerys has a discussion with the Seven Kingdoms' governors but they don't seem interested.
---
Bran leaves Castle Black and takes Arya north. Jon is attacked by a group of Wildlings and saves Bran's life by striking a nerve with him. Tyrion decides to take a different route through the wild places to escape. Theon is captured.
---
Cersei asks Jaime to stay in Winterfell with her, but he says Tyrion has other plans and that Sansa must accompany Jaime. Theon returns to King's Landing after being attacked by the wildlings. At the Eyrie, Joffrey and Varys try to get Jorah the Dreadfort ready for a siege.
---
Tyrion is caught by surprise by an unexpected visitor at the Nightfort, his first, and his first since escaping King's Landing. He is surprised by Jaime's presence and he immediately asks for his leave. Jaime refuses. When Tyrion is about to be imprisoned, he is saved by Podrick. Cersei tries to convince Tyrion to accept the pardon but he declines the request. Jaime and Sansa decide to marry and Tyrion marries Margaery.
---
Tyrion, Daenerys, and Rickon meet with the High Sparrow for advice on the future; the Lannister king asks Cersei and her dragons to help him reclaim the Iron Islands; Stannis confronts the Ironborn on the road; the Seven Kingdoms have an uneasy truce.
---
Arya Stark receives news that the Lannister army has defeated and captured the Ironborn. She also finds Joffrey, Tywin and Littlefinger dead. At Castle Black, a few loyal members of the Watch are being murdered by Stannis Baratheon.
```


## Acknowledgments
Some functions and code snippets were copied from https://github.com/huggingface/transformers
 