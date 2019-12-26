# TV Show Episode Summary Generator

With the code in this repo, it is possible to scrape IMDb and Wikipedia to acquire a large number of episode 
summaries for a TV show, and to use the data to train a GPT-2 model to generate similar summaries.


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
Running the IMDb spider is fairly simple. It only takes a set of keywords as input params: 

Examples:

- Run ```python3 run_imdb_spider.py --title_keywords star trek``` to download the necessary
 data from all the [Star Trek](https://en.wikipedia.org/wiki/Star_Trek)
TV shows from IMDb into a JSON file.

- Run ```python3 run_imdb_spider.py --title_keywords walker texas ranger``` to 
get the episode data for the show [Walker, Texas Ranger](https://en.wikipedia.org/wiki/Walker,_Texas_Ranger).


##### Running the Wikipedia spider
The Wiki spider is slightly more complicated than the previous one. The user have to provide a starting page 
for the recursive search in Wikipedia, plus a string that will be used to filter out URLs 
(URLs that do not contain the substring will be skipped) and a list of keywords,
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
Call ```python3 train.py``` to train a GPT-2 network and save the weights. The script splits the data into a training 
and validation subset. During the training, there is a checkpoint at every X step. At these checkpoints, the loss on 
the validation subset is calculated and a few samples are generated for the user to further monitor the progress.
The best model from the training is saved after the process is finished.

The default params were selected based on what worked best for Star Trek episodes, so it might be a good idea to 
play around with the hyper-parameters a little bit if you are running the training on a different dataset. 
Although, in my experience, the results are satisfactory even when training on a different set of data.

I ran my experiments on a GPU with ~16 GB memory. If you do not have the same resources, you will probably need to 
decrease the batch size or select a smaller GPT-2 model.

Multi-GPU training is currently not implemented.

For more information, check ```python3 train.py -h```.

### Generating episode summaries from scratch
After you have a trained model, you can generate episodes with
```generate.py --num_samples <NUMBER_OF_SAMPLES_TO_GENERATE> --random_seed <SEED>```.

If you want to play around with other parameters of the generation process, check
```python3 generate.py -h```.

If you changed the GPT-2 version from the default ```gpt2-medium``` in the training, you will also have to changed 
it for the generation. 

Example output for Star Trek:
```
When Captain Kirk has a vision of a Vulcan ship, his brain appears on a holographic screen and he starts hallucinating. When Kirk and Spock return to their ship, they find out they had been trapped in a holodeck program.
---
Enterprise visits the planet Tarsus IV to deliver a special gift to the planet's residents, all living in a world that is being destroyed by a mysterious plague; however, it turns out that the plague has already taken effect and only a single-mindedly driven Tardok can save the planet from destruction. Kirk and Spock find a cure but, in the meantime, Tarsusians have to fight to survive and their civilization must be brought back to health by using the ship.
---
A new technology allows B'Elanna Torres to return home in an alternate future and live through her time as captain of Voyager.
---
While traveling across the galaxy, Picard discovers there is a race of people that have been wiped out by time traveling and are waiting for a cure.
---
When the Enterprise travels to the planet Beta IV in search of a lost Romulan vessel, it gets a distress call from Commander Chakotay and Data. They are seeking Captain Jean-Luc Picard, the first Romulan to walk the Earth. The Romulans have just been captured during a joint Romulan-Nistrim action and are now held by the Klingons. In their first attempt to negotiate, the Federation is to release Picard, a man of honor, but not courage, by giving him a free choice of surrender or fighting with the Klingons.
---
When a planet's sun explodes, it leaves behind an alien life force that has a way of killing all who inhabit it.
---
An accident in an asteroid field forces the Enterprise to return to an earlier version of Earth for repairs. Guest star Kevin Smith as Captain Archer.
---
Kirk goes missing while pursuing the Xindi; Picard gets captured and becomes their captive.
---
The Enterprise is sent back in time to Janeway's time at Starfleet Academy, aboard Janeway's old ship, to prevent a planet from being destroyed by a rogue alien faction. The only survivors of the Enterprise are the captain's wife and seven of the crew's children.
---
Odo's father was killed during a battle on the Bajor, so Sisko takes his son to school on Bajor, hoping he will grow up to be a real hero for his people.
---
An asteroid threatens destruction from space; a Federation vessel must find a way to save it.
---
In a rare moment of clarity, Kirk and McCoy beam down to the planet Vissian III. The planet's population has been reduced to a skeleton crew. Only three of six adults remain alive: their leader, Zek. He is a master tactician, but is unable to communicate. It's impossible to know how long they'll remain there before falling into a collective delirium and dying. Kirk, Spock and McCoy beam aboard, but are met by their captor: an old Klingon warrior named Garak. He's a member of their Klingon High Council, and he is willing to sacrifice anyone to bring them home. Kirk and his fellow survivors are then taken to Kirk's old ship, a destroyed one-man space station, where Kirk and McCoy are trained to fight Garak.
---
The Enterprise visits a planet in the Gamma Quadrant that has been abandoned by its native inhabitants. Only two living people remain, a girl named Laila who claims she was sent there by a group of Romulans to kill Earth's Romulans, and a young man, a seemingly unstoppable power-monster who has no inhibitions and who must conquer other people to survive.
---
A Bajoran named Jadzia Dax comes on board and starts playing. She gets along well with Quark and Kira but she is rather hostile when he tries to make a bit more of a deal with her. She's not a nice girl and seems to relish her role. Sisko and Dax decide to put her to the test and convince the other members to follow through.
---
A group of people are trapped aboard a shuttlecraft, and only one of them seems to survive: Spock Prime.
---
When Riker is captured by the Cardassians, the Enterprise is forced to leave its home system and warp to the planet below. On their way to the planet, the ship is attacked by a group of Cardassians who claim that Riker is a spy for the Cardassians. They believe the Enterprise is now working for the Bajorans and want the crew's heads to be cut. Picard, Geordi and La Forge attempt to find out what the Bajorans actually want, but soon realize that the Cardassians have a bigger problem.
```

Example output for Walker, Texas Ranger:
```
Walker makes an unexpected visit to the home of a high school football coach who is accused of murder and must defend himself before a judge. He learns that the coach's wife has been kidnapped by her brother and the two must protect each other.
---
Walker must learn about a group of mercenaries who work for a Russian gang. After a Russian is wounded during a robbery, the group leader demands that his son receive medical treatment, but Walker refuses. The gang leader continues to demand for ransom, which the Rangers are forced to back down from by the demands of their leader.
---
A man with a gun is killed by a Ranger. Walker takes in the man's survivors who want the Ranger back.
---
Walker's cousin, Alex, returns home after a ten-year absence. She is attending to her sister at the hospital after her mother dies in childbirth. Alex is attending to Gage, who recently passed away from liver failure, and is having difficulty in feeding herself. When she tries unsuccessfully to feed herself, Gage gives her a bottle of water and a baby-sitter. The baby is not there, which makes Gage take her back to the hospital and find the baby.
---
Walker and Trivette help a man whom he believes to be a cop-killer, who's actually an enforcer for an armored trucking group. Meanwhile, Alex, Trivette, and Trivette's friend Gage are being tracked by a group of men who are interested in killing a local preacher, who is seen as a symbol of peace.
---
Walker, Trivette, Trivette's brother Carlos and their friend Tommy are all in a small town to help restore order after an armed group of criminals kills everyone from the sheriff to the town's chief. When Tommy witnesses a group of white-supremists murdering a young woman and her husband, he tries to stop them. However, the men are determined to kill Tommy and the girl. So Walker, Carlos, Tommy, Alex and Trivette head to the site of the murder and learn that the white supremacists are after someone in the town who helped them kill the boy's parents.
---
Walker and the Rangers track down a group of men who have stolen a shipment of genetically engineered drugs and are on the run for murder. But when the truck's driver tries to take the drugs in the backseat, things go awry, and the truck is forced to crash in a ravine. Now the Rangers must use their skills to get the truck and its driver before the rest of the stolen truck is damaged beyond repair.
---
Walker tries to save some kids from gang life and gets caught up in the chaos when the leader of a gang is injured, forcing him to flee with some of his gang members. A new leader takes over the gang, the Walker tries to save some of them, but soon finds himself in a fight against the gang and finds himself outmatched for his abilities; his friends get him out of it.
---
Walker tries to keep Trivette away from the Rangers. He asks Ranger Alex, Walker's girlfriend, to take him out to dinner. Walker is having trouble with some old girlfriends. He is upset and asks Trivette to talk about it with him. He tells her that she doesn't know what it means to have a boyfriend and they get back in the car to go. They drive down the highway and stop for gas. One of the guys tries to rob the couple. The girlfriend stops them and says that she knows what it means to be the one who gets them.
---
Walker's father is killed when they are hit by a car while he was on vacation. Walker's brother, Danny, who has a gambling problem, sets his sights on Walker. Alex tries to help Walker but he decides to stay away from the family for personal reasons.
---
Walker learns about Victor Victor Walker's friend Alex's murder in a hotel room with other friends. The young woman is in tears after seeing her murdered lover. Alex must find a way to prevent Victor's killer from testifying in a trial where she has been charged with the crime.
---
Walker and Trivette go undercover on a drug operation run by a gang led by C.D. "Chico" Caswell. The plan goes awry when Caswell learns that Walker knows who he is.
---
Walker's wife, Alex, has gone missing. Her husband's friends believe her to be the reincarnation of a Native American. Walker sets out with Alex's twin brother, Trent, to find her. And soon he becomes trapped in the Cherokee reservation. Trent and Walker must help a Cherokee reservation cop, who has been kidnapped by a group of Indians seeking revenge on the Cherokee for what they consider a land deal they made during the 1857 Apache War.
---
A man who is a former member of the Irish Rangers is working with the Mexican Drug gang. Walker helps him track down and kill several suspects. But when he witnesses a shootout between the mob and Rangers, the gang leader decides to take matters into his own hands.
---
A group of children are kidnapped during a rodeo, and Walker must find their owners and locate their kidnapped cousins.
---
Walker's mother, who works as a nursing home social worker, is killed by a homeless man. Walker learns that he was shot by the man who killed his brother. The man's son then sets out to find the man who shot him.
```

## Acknowledgments
Some functions and code snippets were copied from https://github.com/huggingface/transformers
 