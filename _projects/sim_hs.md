---
layout: distill
title: 'Hearthstone: Euro To In-Game Currency Conversion'
date: 2023-03-08
description: 'With an introduction, that explains the fundamentals of the card game Hearthstone, this article simulates the dust value from opening one pack of cards. This makes it possible to quantify what a Hearthstone deck is worth in Euro.'
img: 'assets/img/stats-sexy.webp'
tags: ['hearthstone', 'money-to-in-game-currency', 'research', 'simulation', 'statistics']
category: ['statistics']
authors: 'Tobias Klein'
comments: true
---
<br>
<br>
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#abstract">Abstract</a></div>
    <div class="no-math"><a href="#research-questions">Research Questions</a></div>
    <div class="no-math"><a href="#introduction">Introduction</a></div>
    <div class="no-math"><a href="#basic-principles-of-the-game">Basic Principles of the Game</a></div>
    <div class="no-math"><a href="#the-odds">The Odds</a></div>
    <div class="no-math"><a href="#calculation-of-odds">Calculation of Odds</a></div>
    <div class="no-math"><a href="#money-to-dust-value-function">Money To Dust Value Function</a></div>
    <div class="no-math"><a href="#simulation-of-dusting-hearthstone-cards">Simulation Of Dusting Hearthstone Cards</a></div>
    <div class="no-math"><a href="#research-question-1---conclusion">Research Question 1 - Conclusion</a></div>
    <div class="no-math"><a href="#research-question-2---conclusion">Research Question 2 - Conclusion</a></div>
    <div class="no-math"><a href="#research-question-3---conclusion">Research Question 3 - Conclusion</a></div>
    <div class="no-math"><a href="#research-question-4---conclusion">Research Question 4 - Conclusion</a></div>
    <div class="no-math"><a href="#research-question-5---conclusion">Research Question 5 - Conclusion</a></div>
  </nav>
</d-contents>

# Hearthstone: Euro To In-Game Currency Conversion

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/stats-sexy.webp" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
       Just as many other games, Hearthstone uses a <i>loot box</i> system
       to monetize the game. The player can only buy card packs for the most
       part. These rarely contain the card(s) that the player actually wants.
       There is however one universal in-game currency. The dust value, that
       each of the rarity classes has. Using hacker statistics, this article
       calculates the relationship between Euro spent and equivalent in-game
       dust value.
</div>

## Abstract

This article calculates the equivalent average dust value (ADV) of a single
Hearthstone card pack. Specifically, this project aims to define a function
whose independent variable is Euro spent. The dependent variable of this
function is the equivalent dust value one can expect, given the amount spent.
There is a universal in-game currency called dust value, which indicates how
expensive decks are. Two different simulations for that purpose are evaluated
and compared in terms of approximate runtime and results. It is shown that the
ADV converges towards a Normal distribution with identical mean and standard
deviation values. It follows that the here found results are in agreement with
the central limit theorem. The implications of the Law of Large Numbers for this
study's findings are discussed as well.

## Research Questions
1\. Compare the performance of two different simulation functions written in
   Python and analyze the results.
 - A vectorized function without loops (all numpy)
 - A loop based function (some numpy, but no vectorization)

2\. What is the ADV per card pack?<br>
3\. Show that the distribution of the ADV converges against a normal
distribution $$N(\mu_{N},\sigma_{N})$$ with $$\mu_{N}\,=\hat{\mu}$$ and
$$\sigma_{N}\,=\,\hat{\sigma}$$, $$\hat{\mu}$$ and $$\hat{\sigma}$$ being the
empirical mean and standard deviation from the simulation, as stated by the
central limit theorem.<br>
4\. Explain how the *Law of Large Numbers* applies here.<br>
5\. How is the function defined that maps Euro spent to the equivalent dust
value.

<!-- simulate the process of opening card packs in the -->
<!-- game of Hearthstone. -->

<!-- Further, it tries to estimate the average dust value (**ADV**) of a card -->
<!-- pack.  in terms of how much money has to be spent buying card packs, in order for a player who does -->
<!-- not have the cards that make up the new deck to create it. to -->
<!-- create one of the Tier 1 decks. *Casual player* here refers to someone who does -->
<!-- not have all cards -->

## Introduction

<br>
From the Wikipedia entry:

> Hearthstone is a free-to-play online digital collectible card game developed
> and published by Blizzard Entertainment. Originally subtitled Heroes of
> Warcraft, Hearthstone builds upon the existing lore of the Warcraft series by
> using the same elements, characters, and relics.
> [Wikipedia](https://en.wikipedia.org/wiki/Hearthstone)

<br>

Hearthstone is a popular trading card game that was released by [Activision
Blizzard](https://www.activisionblizzard.com/) in 2014. In 2017, even before the
surge in player base caused by the lockdown during the COVID-19 pandemic, the
game [was reportedly generating as much as $$40 million a month](https://venturebeat.com/pc-gaming/pc-gaming-weekly-watch-out-heartstone-here-comes-artifact/).

*The table shows various statistics regarding the player base of the game
Hearthstone. It can be said that according to this table, there are around 4
million monthly players on average. Data originates from
[https://activeplayer.io/hearthstone/](https://activeplayer.io/hearthstone/) and
is only reproduced here.*

<div id="table_2_wrapper" class="wpDataTables wpDataTablesWrapper no-footer wpDataTableID-38 wpdt-pagination-right"><div class="dt-buttons"></div><div class="clear"></div><table id="table_2" class="display nowrap data-t data-t wpDataTable wpDataTableID-38 dataTable no-footer" style="" data-described-by="table_2_desc" data-wpdatatable_id="38" role="grid">
        <!-- Table header -->
<thead>
<tr role="row"><th data-class="expand" class="wdtheader sort expand sorting_disabled column-month" style="" rowspan="1" colspan="1">Month</th><th class="wdtheader sort numdata integer sorting_disabled column-average-monthly-players" style="" rowspan="1" colspan="1">Average Monthly Players</th><th class="wdtheader sort numdata integer sorting_disabled column-monthly-gain--loss" style="" rowspan="1" colspan="1">Monthly Gain / Loss</th><th class="wdtheader sort numdata float sorting_disabled column-monthly-gain--loss-" style="" rowspan="1" colspan="1">Monthly Gain / Loss %</th><th class="wdtheader sort numdata integer sorting_disabled column-max-players-in-a-day" style="" rowspan="1" colspan="1">Max Players in a Day</th></tr>
</thead>
        <!-- /Table header -->
        <!-- Table body -->
<tbody><tr id="table_38_row_0" role="row" class="odd">
                            <td style="" class="  column-month">Last 30 Days</td>
                            <td style="" class=" numdata integer  column-average-monthly-players">4,110,654</td>
                            <td style="" class=" numdata integer  column-monthly-gain--loss">8,398</td>
                            <td style="" class=" numdata float  column-monthly-gain--loss-">1.11</td>
                            <td style="" class=" numdata integer  column-max-players-in-a-day">342,555</td>
                    </tr><tr id="table_38_row_1" role="row" class="even">
                            <td style="" class="  column-month">March 30, 2022</td>
                            <td style="" class=" numdata integer  column-average-monthly-players">4,102,256</td>
                            <td style="" class=" numdata integer  column-monthly-gain--loss">36,754</td>
                            <td style="" class=" numdata float  column-monthly-gain--loss-">5.45</td>
                            <td style="" class=" numdata integer  column-max-players-in-a-day">341,855</td>
                    </tr><tr id="table_38_row_2" role="row" class="odd">
                            <td style="" class="  column-month">February 28, 2022</td>
                            <td style="" class=" numdata integer  column-average-monthly-players">4,065,502</td>
                            <td style="" class=" numdata integer  column-monthly-gain--loss">175,444</td>
                            <td style="" class=" numdata float  column-monthly-gain--loss-">1.22</td>
                            <td style="" class=" numdata integer  column-max-players-in-a-day">338,792</td>
                    </tr><tr id="table_38_row_3" role="row" class="even">
                            <td style="" class="  column-month">January 30, 2022</td>
                            <td style="" class=" numdata integer  column-average-monthly-players">3,890,058</td>
                            <td style="" class=" numdata integer  column-monthly-gain--loss">-126,543</td>
                            <td style="" class=" numdata float  column-monthly-gain--loss-">-0.54</td>
                            <td style="" class=" numdata integer  column-max-players-in-a-day">324,172</td>
                    </tr><tr id="table_38_row_4" role="row" class="odd">
                            <td style="" class="  column-month">December 30, 2021</td>
                            <td style="" class=" numdata integer  column-average-monthly-players">4,016,601</td>
                            <td style="" class=" numdata integer  column-monthly-gain--loss">105,546</td>
                            <td style="" class=" numdata float  column-monthly-gain--loss-">0.13</td>
                            <td style="" class=" numdata integer  column-max-players-in-a-day">334,717</td>
                    </tr></tbody>        <!-- /Table body -->
 <!-- Table footer -->
        <!-- /Table footer -->
    </table></div>


## Basic Principles of the Game

### Format
1. Every match of *traditional* Hearthstone, which is the focus of this
   article, is a 1 vs. 1 match where two players face each other. Each player has a
   deck of cards consisting of 30 cards. This article is not concerned with how
   the actual game works and only explains the details needed for this article. For
   more detailed information regarding the basic game mechanics, please see
   [this official
   guide](https://hearthstone.blizzard.com/en-gb/how-to-play?topic=innkeepers-lessons)
   for example.

### Deck Of Cards
2. Each player has to have at least one *deck* in order to be able to play.
    - A deck is a collection of 30 cards.
    - The cost of a deck is equal to the sum of the *dust costs* of all 30 cards in the deck.
    - This assumes that
        the player has none of the 30 cards in his card collection that make up
        the deck.
    - Depending on the distribution of the rarities of the 30 cards
        (more on that later), a deck can cost between 4,000 to 18,000 *dust* on
        average. There are generally many decks that can be found somewhere
        between these limits.
    - These limits apply to decks that do not have
        golden cards in them. A deck that contains golden cards costs between 4
        to 9 times more than a deck without golden cards.

### Ways To Acquire Cards
3. There are several ways one can acquire the digital cards necessary to create
   a deck.
   1. Use the in-game currency *coins* to buy packs of cards, which are gained by
      completing quests among other in-game activities. This method will be
      referred to as, *by playing the game*.
   2. Buying card packs with real money. There are several *bundles* on offer
      at the time of writing this article. They are listed in **Table 1**. There
      are special offers from time to time, which will not be evaluated in this
      article.
   3. *Disenchanting* cards see [The Universal In-Game Currency](#the-universal-in-game-currency) that are in one's collection
      and that can be disenchanted. Generally, all cards that can be bought by
      purchasing packs, can also be disenchanted.

<style type="text/css">
#pt {
  margin-left: auto;
  margin-right: auto;
}
#pt {
caption-side: top;
font-size: 80%;
}
</style>
<table id="pt" class="center">
<caption><strong>Table 1</strong> shows the purchase options available in the game of
Hearthstone.</caption>
<thead>
  <tr>
    <th class="tg-fymr">Number of Packs in Bundle</th>
    <th class="tg-fymr">Price (Eur.)</th>
    <th class="tg-fymr">Normalized Price (per pack, rounded)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">2</td>
    <td class="tg-0pky">2.99</td>
    <td class="tg-0pky">1.5</td>
  </tr>
  <tr>
    <td class="tg-0pky">7</td>
    <td class="tg-0pky">9.99</td>
    <td class="tg-0pky">1.43</td>
  </tr>
  <tr>
    <td class="tg-0pky">15</td>
    <td class="tg-0pky">19.99</td>
    <td class="tg-0pky">1.33</td>
  </tr>
  <tr>
    <td class="tg-0pky">40</td>
    <td class="tg-0pky">49.99</td>
    <td class="tg-0pky">1.25</td>
  </tr>
</tbody>
</table>


### The Universal In-Game Currency
4. Dust is the universal currency in the game for calculating the actual cost
   associated with acquiring/crafting (creating) any card and ultimately deck.
   1. There are **4 classes** of cards and each class has a golden variant. They
      are:
      - common and golden common
      - rare and golden rare
      - epic and golden epic
      - legendary and golden legendary
   2. The **Golden variants** are rarer compared to their non-golden
      counterparts. They cost multiple times more *dust* to create compared to
      their non-golden counterparts, but also give more *dust* when *disenchanted*.
   3. **Disenchanting** means deleting the card from one's collection in exchange
      for dust. See table 2, for the exact *dust* values for each rarity class.

### A Pack Of Cards
5. Each pack of cards has 5 cards. Each card can be of any rarity (See Table 2
   for the names of the classes) in general. There is one exception to this, if
   4 out of the 5 cards are of the lowest rarity class (Common). In this case,
   the 5th card will be at least of *Rare* quality, the second lowest rarity
   class in terms of *drop chance* and *dust* value. This mechanism is not
   relevant for this simulation however, since only complete sets of *drop
   chances* that cover all rarity classes are considered. More on *drop chances*
   in the following.

## The Odds

Hearthstone and many other computer games rely on a **loot box** system for
monetization. Its aim is to make the player either spend time earning *coins*
by playing the game or/and money to acquire new cards by buying packs of cards.
The *loot box* system relies on in-game items that give the player loot, cards
from card packs in this instance. The set of probabilities (*draw chances*)
associated with how likely it is to draw a card of a certain rarity class when
opening a pack of cards (see table 2 for exact values), does not have to be
disclosed by Blizzard. There is only China as of now that demands what seems
like a partial disclosure of the *drop chances* for any video game company
that wants to enter the Chinese market with their video game. See statement
below for more.

### What We Know

#### Original Statement (Chinese)

>  关于《炉石传说》卡牌包抽取概率的公示方式调整公告
> 发布日期：2018-08-02
> 《炉石传说》现将抽取卡牌的概率公示方式进行调整，具体如下：
> 《炉石传说》卡牌包共有5张卡牌，包含4种不同品质(普通、稀有、史诗、传说)
> 稀有卡牌
> 每个炉石卡牌包，至少能获得一张稀有或更高品质的卡牌。即100%的卡牌包可至少开出稀有或更高品质（史诗、传说）的卡牌。
> 史诗卡牌
> 约20%的卡牌包可开出史诗品质卡牌。
> 传说卡牌
> 约5%的卡牌包可开出传说品质卡牌。
> 备注：
> - 每个账号首次打开10包同一种类卡牌包（如经典、女巫森林、砰砰计划等），必定可开出传说品质卡牌。
> - 本次公告调整仅涉及概率公示方式的调整。《炉石传说》卡牌包的实际抽取概率并未发生改变。

<br>
<br>

#### English Translation

> Announcement on the adjustment of the disclosure method of the card pack extraction probability of "Hearthstone Legend
> Release date: 2018-08-02
> The Legend of Hearthstone is now adjusting the public announcement method of the probability of card extraction, as follows.
> Hearthstone Legend card pack has 5 cards, including 4 different qualities (common, rare, epic, legend)
> Rare cards
> Each Hearthstone card pack, you can get at least one rare or higher quality card. That is, 100% of the packs can open at least rare or higher quality (epic, legendary) cards.
> Epic Cards
> About 20% of card packs open Epic quality cards.
> Legendary Cards
> Approximately 5% of the packs will yield Legendary quality cards.
> Remark.
> - The first time each account opens 10 packs of the same kind of cards (such as Classic, Witch's Wood, Bang Bang Project, etc.), it will definitely open Legendary quality cards.
> - This announcement only involves the adjustment of the probability announcement method. The actual probability of drawing cards from The Legend of Hearthstone packs has not changed.

<br>
<br>

The statement from Blizzard reveals only some of the drop chances associated
with opening card packs. What can be gained from the statement is the following:

- At least one card in any pack will be of class *rare* or higher.

  This fact is not of relevance for this case study, as discussed in section
  'Basic Principles of the Game, Bullet Point 5\.')

- Cards of the class *epic* have a 20% drop chance.

  This is valuable, since it gives a presumably accurate drop chance for *epic*
  cards. It is not a number that comes from empirical testing, but from the
  creator of the draw mechanism itself. It is therefore used in the following.

- *Legendary* cards have a drop chance of 5%.

  As with the drop chance for *epic* cards, this number is used in the
  following. See *epic* card drop chance for the reasoning.

## Calculation of Odds

### Sources
With the drop chances for *epic* and *legendary* cards from the afore mentioned
Blizzard statement and various empirical studies where people opened packs and
counted the cards they had opened during the study
([source](https://hearthstone.fandom.com/wiki/Card_pack_statistics#Expected_dust_value)),
as references, the drop rates were calculated as follows.

### Sorting By Rarity
- All empirical studies showed that the drop chances follow an order. From
  least to most likely, for all classes:
        1. Golden Legendary
        2. Golden Epic
        3. Golden Rare
        4. Golden Common
        5. Legendary
        6. Epic
        7. Rare
        8. Common

### Limitations
It must be noted that all empirical studies were conducted before 2018.
Blizzard's statement is the most recent information, and it is presumed that It
is valid for sales made outside of China as well. Empirical experiments
conducted on the distribution of the rarity classes mentioned in the Blizzard
statement did not match those that Blizzard disclosed in the statement above, as well
as because the sum of probabilities for all classes wasn't exactly 1.0. So some
adjustments had to be made, to arrive at a set of probabilities that is
consistent with Blizzard disclosed numbers where the sum of probabilities is
exactly 1.0 as well.

### Final Odds
The final values for the drop chances used, come from the sources mentioned
earlier and from testing different sets of drop chances, in order to rule out
sets that either violate the order outlined or don't sum up to exactly 1 or
violate the information given by Blizzard.  The final, *correct* set of drop
chances remains unknown, but using the simulation approach outlined in the
following, any valid set can be tested and its results analyzed. The drop
chances can be found in table 2.

<br>
<br>

**Table 2** *The columns of Table 2 show, from left to right: All card classes
in the 'Class' column. The 'Draw Chance' column gives the drop chances for each
class and 'Dust Value' gives the amount of dust that disenchanting the
respective class gives.*

| Class            | Draw Chance | Dust Value |
|:-----------------|:-----------:|:----------:|
| golden legendary |    0.0007   |    1600    |
| golden epic      |    0.0023   |     400    |
| golden rare      |    0.0133   |     100    |
| golden common    |    0.0137   |     50     |
| legendary        |     0.05    |     400    |
| epic             |     0.2     |     100    |
| rare             |     0.25    |     20     |
| common           |     0.47    |      5     |

## Money To Dust Value Function

With $$m$$ the money spent, $$n$$ the number of packs bought. $$n$$ can be any
linear combination of the number of packs available, as shown in Table 1. $$p$$
is the price for $$n$$ packs. $$\mathit{ADV}$$ is the dust average per pack from
the simulation.

$$
\def\lf{\left\lfloor}
\def\rf{\right\rfloor}
f(m,n,p,\mathit{ADV})= \lf \frac{m}{p} \rf \cdot \mathit{ADV} \cdot n
$$

The result is a linear function, with a bias of 0 and gradient larger 1 for all
possible values of $$n$$. The steepest gradient is found for multiples of the
largest card pack bundle with 40 packs in each bundle.

## Simulation Of Dusting Hearthstone Cards

The term 'dusting' here refers to the act of *disenchanting* hearthstone cards.
During the process of dusting a card, the card becomes permanently deleted from
one's card collection. In return, one gets the dust value, that the card is
worth. Please refer to *Table 2* for further details.

### Imports


```python
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState, seed
import pandas as pd
import time

plt.style.use("science")
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

seed(42)
plt.ion()
plt.close("all")
```

### Numpy Vectorization Implementation (NVI)

#### Inputs

The following is a much better way to define the numbers needed for the
simulation than is used in the original loop based implementation, described
next. With a dictionary, the relevant input values are defined for the
simulations. `trials` specifies the number of trials to simulate, each with the
number of packs `packs`. `probs` gives the draw chances for each class. `cats`
is used to transform the string rarity classes into integer ones. `dust` gives
the dust value for each rarity class.


```python
# order draw_chances values to create intervals for drawing mechanism
inputs = {
    "trials": np.int_(1e5),
    "packs": np.int_(40),
    "probs": {
        "glegendary": np.double(0.0007),
        "gepic": np.double(0.0023),
        "grare": np.double(0.0133),
        "gcommon": np.double(0.0137),
        "legendary": np.double(0.05),
        "epic": np.double(0.2),
        "rare": np.double(0.25),
        "common": np.double(0.47),
    },
    "cats": {
        "glegendary": 7,
        "gepic": 5,
        "grare": 3,
        "gcommon": 1,
        "legendary": 6,
        "epic": 4,
        "rare": 2,
        "common": 0,
    },
    "dust": {
        "glegendary": np.int_(1600),
        "gepic": np.int_(400),
        "grare": np.int_(100),
        "gcommon": np.int_(50),
        "legendary": np.int_(400),
        "epic": np.int_(100),
        "rare": np.int_(20),
        "common": np.int_(5),
    },
}
```

#### NVI Function Definition

Using the default numpy rng generator, with output values from `dust`,
probabilities `probs` and size the product of `trials`, `packs` and 5. There are
5 cards in each card pack. Using `cards.reshape(trials,packs,5)` to create a
ndarray with suitable dimensions, the ADV over each trial is calculated and
returned along with the standard deviation for the distribution of the ADV.



```python
def sim_rc(
    packs=inputs["packs"],
    trials=inputs["trials"],
    probs=[*inputs["probs"].values()],
    dust=[*inputs["dust"].values()],
):
    rng = np.random.default_rng()
    s = np.product([packs, trials, 5])
    cards = np.array(rng.choice(a=dust, size=s, p=probs))
    sima = cards.reshape(trials, packs, 5)
    dust_avg_t = [np.sum(tt) / packs for tt in sima]
    dust_avg = np.mean(dust_avg_t)
    dust_std = np.std(dust_avg_t)
    return dust_avg_t, dust_std, dust_avg
```

#### NVI Call Function



```python
start_np = time.time()
np_mu_trial, np_std, np_mu = sim_rc()
total_time_np = time.time() - start_np
print(
    f'The system time duration, total cards generated for the NVI is: {total_time_np}, {inputs["packs"]*inputs["trials"]*5:,}'
)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax = plt.subplot(111)
ax.hist(np_mu_trial, density=False, bins=200, label="histogram", alpha=1)
plt.show()
```

    The system time duration, total cards generated for the NVI is: 0.8454718589782715, 20,000,000



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects_sim_hs/output_16_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 1: First look at the absolute frequency of the ADV value, as
        produced by the NVI.
</div>
    



```python

print(
    f"The sorted drop chances for the classes are: {[*zip([*inputs['cats'].values()],[*inputs['probs'].values()])]}"
)
print(f"The sum of the sorted drop chances is: {sum([*inputs['probs'].values()]):.4f}")
```

    The sorted drop chances for the classes are: [(7, 0.0007), (5, 0.0023), (3, 0.0133), (1, 0.0137), (6, 0.05), (4, 0.2), (2, 0.25), (0, 0.47)]
    The sum of the sorted drop chances is: 1.0000


### Loop Implementation (LI)

*This implementation should not be used, it is much slower than the NVI using
only numpy vectorization. It is only included for comparison and learning
purposes.*

#### Motivation *Optional*
This is the first implementation I wrote back in 2019, when I was still actively
playing Hearthstone and wanted to know how much money I had to spend (on
average), in order to be able to craft a 'Golden Zephrys the Great' (Cost: 3200
dust). The card is special, as it uses machine learning, to understand what the
best three cards are for the player at the time of the card being played. It
then offers the player these three cards and plays the one chosen. In Blizzard's
words, the model had to be changed, so its predictions were not as refined and
'imbalanced' anymore.<br>
![Golden Zephrys the Great](https://cards.hearthpwn.com/enUS/gifs/ULD_003.gif?165029)
<br>

In short, they nerfed the card soon after its addition to the game.
Interestingly, the machine learning model behind this mechanic was sophisticated
enough that even Blizzard said that they could not always forecast what the
three chosen cards would be, as the models predictions in the form of the three
'perfect' cards presented to the player were the result of some sort of 'black
box' type of model. My guess is that it is a deep learning model.

Anyway, after this brief motivation of why I embarked on the project of
simulating the 'opening of hearthstone card packs' in the first place, I present
the original loop implementation.

#### LI Inputs
A discrete distribution with intervals for each rarity class on the domain
$$[0,1]$$ is used. The intervals are constructed, so that the area of each
interval is equal to the respective drop chance of the rarity class it
represents.

Each interval is defined by the lower and upper limit of the half open interval
between `_l` and `_h` ($$(\,\_l,\_h]\,$$) for all rarity classes, like:
`gl_l`=*goldenlegendary_low* and `gl_h`=*goldenlegendary_high* for example.
Starting by zero, the intervals are added to the upper limit of the previous
rarity class from the class with the lowest draw chance up to the class with the
highest. The sum of the intervals must equal exactly 1 to be a valid set of
probabilities.



```python
# creating interval borders: '_l'/'_h' stand for the low end and high end of the interval respectively.
gl_l = 0.0
gl_h = 0.0007
ge_l = gl_h
ge_h = gl_h + 0.0023
gr_l = ge_h
gr_h = ge_h + 0.0133
gc_l = gr_h
gc_h = gr_h + 0.0137
l_l = gc_h
l_h = gc_h + 0.05
e_l = l_h
e_h = l_h + 0.2
r_l = e_h
r_h = e_h + 0.25
c_l = r_h
c_h = r_h + 0.47

print(
    f"The probability of a common class card to be drawn is {np.round(1-c_l,decimals=2)}"
)

# Defining rarity colors by integer numbers
rarity_dict = {
    "common": 0,
    "golden_common": 1,
    "rare": 2,
    "golden_rare": 3,
    "epic": 4,
    "golden_epic": 5,
    "legendary": 6,
    "golden_legendary": 7,
}
```

    The probability of a common class card to be drawn is 0.47


#### LI Function Definition

A numpy function is used to generate random numbers from a uniform distribution
on the domain $$[0,1)$$. Half open intervals used for the probability account
for the domain of the numpy function used. The product of packs per trial, trial
and 5 (there are 5 cards in each pack) gives the total number of random numbers
it generates. `np.nditer` is used to create an iteratee over which then is
iterated. The outer most loop over `onet` loops over each trial. With the next one
looping over each pack opened in each trial (`draws_onet`). The inner most loop
then loops over the opening of the five cards in every pack (`rows[:]`).

For each random number that represents one drawn card in a pack of five cards,
it is evaluated which of the intervals it lies within and the corresponding
rarity class and dust value is appended to a list. There are lists for each
loop and lists `all_packs`, `all_dust` hold these values for the entire
simulation.



```python
def hs_sim(packs, trials):
    """Function that simulates opening Hearthstone card packs. Each card drawn
    is logged, including its rarity class and dust value."""

    print(f"Simulating {trials:,} trials, each with {packs} card packs")
    all_draws = np.random.uniform(size=packs * trials * 5)
    all_draws = all_draws.reshape(trials, packs, 5)
    # create iterator over trials axis.
    onet = np.nditer(np.arange(0, trials, 1))
    all_packs, all_dust = [], []
    # loop over each trial.
    for i in onet:
        # Only keep i-th trial index value and all index values for 2nd,3rd
        # axes.
        draws_onet = all_draws[i]
        # reset trial stats at start of every trial.
        dust_onet, cards_onet = [], []
        # loop over each pack in the trial with all 5 column values in each row.
        for row in draws_onet:
            # reset values for pack contents after 5 cards have been opened.
            # 5 cards per pack to open.
            cards_onep, dust_onep = [], []
            # For each opened card, identify it and add corresponding values for
            # type and dust value to cards_onep and dust_onep.
            for draw_result in row[:]:
                card_single, dust_single = -999, -900
                # 1 golden legendary
                if draw_result <= gl_h:
                    card_single, dust_single = 7, 1600
                # 2 golden epic
                elif ge_l < draw_result <= ge_h:
                    card_single, dust_single = 5, 400
                # 3 golden rare
                elif gr_l < draw_result <= gr_h:
                    card_single, dust_single = 3, 100
                # 4 golden common
                elif gc_l < draw_result <= gc_h:
                    card_single, dust_single = 1, 50
                # 5 legendary
                elif l_l < draw_result <= l_h:
                    card_single, dust_single = 6, 400
                # 6 epic
                elif e_l < draw_result <= e_h:
                    card_single, dust_single = 4, 100
                # 7 rare
                elif r_l < draw_result <= r_h:
                    card_single, dust_single = 2, 20
                # 8 common
                elif c_l < draw_result <= c_h:
                    card_single, dust_single = 0, 5
                # append dust value of card in pack.
                dust_onep.append(dust_single)
                # append card category of card in pack.
                cards_onep.append(card_single)
            # at the end of each pack opening append all values in dust_onep to
            # dust_onet.
            dust_onet.append(dust_onep)
            # at the end of each pack opening append all values in cards_onep to
            # cards_onet.
            cards_onet.append(cards_onep)
        # at the end of each trial append all values in dust_onet to
        # all_dust.
        all_dust.append(dust_onet)
        # at the end of each trial append all values in cards_onet to
        # all_packs.
        all_packs.append(cards_onet)
    # create ndarrays, with two dimensions (trials,packs*5)
    all_packs = np.array(all_packs).reshape(trials, packs * 5)
    all_dust = np.array(all_dust).reshape(trials, packs * 5)
    # return the two ndarrays
    return all_packs, all_dust
```

#### LI Call Function
The LI function is given the same inputs as the NVI and its duration is logged
as well.


```python
# mm = (40, 100,000  # packs per trial, trials
mm = (inputs["packs"], inputs["trials"])
packs, trials = mm[0], mm[1]

start_for_loop = time.time()
a1, a2 = hs_sim(packs, trials)
total_time_for_loop = time.time() - start_for_loop
print(
    f'The system time duration, total cards generated for the LI is: {total_time_for_loop}, {inputs["packs"]*inputs["trials"]*5:,}'
)
```

    Simulating 100,000 trials, each with 40 card packs
    The system time duration, total cards generated for the LI is: 22.610981941223145, 20,000,000


## Research Question 1 - Conclusion

> 1. Compare the performance of two different simulation functions written in
>    Python and analyze the results.
>    1. A vectorized function without loops (all numpy)
>    2. A loop based function (some numpy, but no vectorization)

### Elapsed Time
The difference in elapsed time between the two functions is enormous. Given that
this is the actual system time, the time elapsed in executing each
implementation can be influenced by the system load on the local machine. There
may be a difference in results from one execution to another. The scalar value $$m$$ that gives the following relationship between the times for the implementations $$\mathit{LI}_{t}\,=\,m \mathit{NVI}_{t}$$ tends to fall within the interval $$m \in \mathbb{R} \cap [25,28]$$

The value of $$m$$ with respect to the most recent execution is printed below.


```python
m = total_time_for_loop / total_time_np
print(f"The scalar using the most recent executions is: {m}")
```

    The scalar using the most recent executions is: 26.743624522935473


### Simulation Size
It takes the LI ~23 seconds to simulate the drawing of 20,000,000 cards. Despite
the fact that the NVI always outperforms the LI, it is the NVI that should be
considered when simulating large numbers of pack openings.

### ADV Value And Distribution Comparison
The final ADV value over all trials is close to the same for both. Not only is
this true for the ADV value, but also for the PDFs of the ADV per trial. See the
histogram below for a comparison of the distributions.


```python
dust_avg_t = [np.sum(tt) / packs for tt in a2]
li_std = np.std(dust_avg_t)
plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax = plt.subplot(111)
ax.hist(
    np_mu_trial,
    density=True,
    bins=200,
    label=f"for-loop | {inputs['trials']:,} trials",
    alpha=0.5,
)
ax.hist(
    dust_avg_t,
    density=True,
    bins=200,
    label=f"numpy-vectorization | {inputs['trials']:,} trials",
    alpha=0.5,
)
ax.set_title(f"Comparsion Of Mean Dust Values")
ax.set_xlabel(
    f"Mean Dust Values For Single Pack, By Trial ({inputs['packs']} Packs Per Trial)"
)
ax.set_ylabel("Absolute Frequency")
ax.legend(loc="best")
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects_sim_hs/output_27_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 2: Plot showing the PDFs of both implementations. The PDF of the
        NVI (<i>red</i>) is overlaid with the PDF of the LI (<i>teal</i>). One can tell
        by the plot, that the two are almost identical. Both are the result of
        100,000 samples, each with 40 cards packs.
</div>
    


## Research Question 2 - Conclusion

> 2\. What is the ADV per card pack?

Empirical results for the mean and standard deviation of $$ADV_{trial}$$ using 40
packs per trial and 100,000 trials are in dust per card pack:

$$
\begin{align*}
\hat{\mu}(ADV_{trial}) &= \{\,\,x \in \mathbb{R}\quad |\quad 256 \,\le \, x \, \le \, 258\,\,\}\\
\hat{\sigma}(ADV_{trial}) &= \{\,\,x \in \mathbb{R}\quad |\quad 34 \,\le \, x
\,\le 36 \,\,\}
\end{align*}
$$

<br>


```python
print(
    f"LI - Average dust per trial of the simulation (first five trials with {inputs['packs']} each):\n {dust_avg_t[:5]}"
)
dust_avg_all_trials = np.mean(dust_avg_t)
print()
print(
    f"LI - ADV and standard deviation of ADV over all trials in the simulation: {dust_avg_all_trials}, {li_std}"
)
print(
    f"NVI - ADV and standard deviation of ADV over all trials in the simulation: {np_mu}, {np_std}"
)
```

    LI - Average dust per trial of the simulation (first five trials with 40 each):
     [282.75, 244.0, 248.875, 242.625, 257.0]
    
    LI - ADV and standard deviation of ADV over all trials in the simulation: 257.0840275, 34.91854175261681
    NVI - ADV and standard deviation of ADV over all trials in the simulation: 256.94153875, 35.060196821873355


Further documentation of simulation results. Unique dust values and their
respective absolute frequencies over the entire simulation, ordered by unique
card categories and indexed with string names of each card category.


```python
un, freq = np.unique(a2, return_counts=True)
keys = [
    "golden_legendary",
    "legendary_golden-epic",
    "epic_golden-rare",
    "golden_common",
    "rare",
    "common",
]
dfu = pd.concat(
    [pd.Series(un), pd.Series(freq)],
    axis=1,
).sort_values(by=0, ascending=False)
dfu.columns = ["unique_dust_categories", "abs_frequency"]
dfu.index = keys
dfu
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_dust_categories</th>
      <th>abs_frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>golden_legendary</th>
      <td>1600</td>
      <td>14092</td>
    </tr>
    <tr>
      <th>legendary_golden-epic</th>
      <td>400</td>
      <td>1046900</td>
    </tr>
    <tr>
      <th>epic_golden-rare</th>
      <td>100</td>
      <td>4263543</td>
    </tr>
    <tr>
      <th>golden_common</th>
      <td>50</td>
      <td>273122</td>
    </tr>
    <tr>
      <th>rare</th>
      <td>20</td>
      <td>5000453</td>
    </tr>
    <tr>
      <th>common</th>
      <td>5</td>
      <td>9401890</td>
    </tr>
  </tbody>
</table>
</div>



## Research Question 3 - Conclusion

> 3\. Show that the distribution of the ADV converges against a normal
distribution $$N(\mu_{N},\,\sigma_{N})$$ with $$\mu_{N}\,=\hat{\mu}$$ and
$$\sigma_{N}\,=\,\hat{\sigma}$$, $$\hat{\mu}$$ and $$\hat{\sigma}$$ being the
empirical mean and standard deviation from the simulation, as stated by the
central limit theorem.<br>

A histogram showing the relative frequency of dust means per trial over the
entire series. Smaller sample sizes, as well as fewer packs per trial, result in
less Bell Curve shaped PDFs, with the PDF for simulations with 40 packs (the
maximum one can buy at a time, at the time of writing) per trial and 100,000
trials per simulation approaching the shape of a Normal Distribution with the
same mean and standard deviation as found in the simulation sample.

Sample means become normally distributed as sample sizes tend to infinity,
according to the Central Limit Theorem. Therefore, it should be converging
towards the population mean standard deviation. The largest sample size, limited
by the local M1 chip apple machine, was 100,000 trials each with 40 packs and 5
cards per pack. In the largest simulation, 20,000,000 cards are drawn. As shown
below, the PDF of the ADV approximates the PDF of a Normal Distribution with the
same mean and standard deviation.

The plot, where the empirical PDFs from the simulations are compared with each
other and with the PDF of the Normal distribution described above. This
distribution has identical mean and standard deviation, the goodness of fit of
the Normal distribution remains somewhat unclear. The plot does show that the
overall bell curve-like shape is shared by all three PDFs. The Normal
distribution uses the mean and standard deviation from LI, which are identical
to the ones from the NVI for all plots.



```python
mu = dust_avg_all_trials
li_std = np.std(dust_avg_t)
dist = norm(mu, li_std)
low_end = int(np.ceil(dist.ppf(0.0001)))
high_end = int(np.ceil(dist.ppf(0.9999)))
x_vals = [x for x in np.arange(low_end, high_end, 0.5)]
y_vals = [dist.pdf(x) for x in x_vals]

print(f"The current mu, sigma have values {mu} and {li_std} respectively.")

fig, ax = plt.subplots(1, 1, figsize=(18.5, 12.5))
plt.title(f"Histogram from packs,trials: {packs},{trials:,}", fontsize=14)
ax.hist(dust_avg_t, density=True, bins=200, label="LI Density Dust per Trial")
ax.hist(np_mu_trial, density=True, bins=200, label="NVI Density Dust per Trial")
ax.plot(
    x_vals,
    y_vals,
    color="#850859",
    alpha=0.7,
    label="PDF-Normal w/ $μ$ and $σ$ from Simulation",
)
# ax.axvline(x=np.min(dust_avg_t), color="y", linestyle="dotted", label="min of μ")
# ax.axvline(x=np.max(dust_avg_t), color="r", linestyle="dotted", label="max of μ")
ax.set_ylabel("Relative Frequency")
ax.set_xlabel("Average Dust Per Trial")
ax.set_xlim(low_end, high_end)
ax.legend(loc="best", fontsize=10)
plt.show()
```

    The current mu, sigma have values 257.0840275 and 34.91854175261681 respectively.



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects_sim_hs/output_33_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 3: The PDFs of both implementations are overlaid and compared to
        the PDF of a Normal distribution with the same mean and standard
        deviation.
</div>
    


The ECDF plots for both implementations plotted together with the CDF of the
theoretical Normal distribution described above, gives a clearer picture as to
where the differences are found between the theoretical Normal distribution and
the inseparable simulation ECDF functions.


```python
# create the ECDF plot for the average dust per trial Random Variable
ecdf = ECDF(dust_avg_t)
ecdf_np = ECDF(np_mu_trial)
# The first value in ecdf.x is -inf, which pyplot does not like.
# So the first value in ecdf.x and ecdf.y is dropped to keep their lengths the same.
ecdf.x = ecdf.x[1:]
ecdf.y = ecdf.y[1:]
ecdf_np.x = ecdf_np.x[1:]
ecdf_np.y = ecdf_np.y[1:]

# using mu and sigma from above as parameters for the theoretical distribution
dist2 = norm.rvs(mu, li_std, 20000000)
dist2s = np.sort(dist2)
ecdf_n = np.arange(1, len(dist2s) + 1) / len(dist2s)
plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)
plt.title(
    f"Theoretical CDF from μ, σ from packs,trials:\n{packs},{trials:,} and ECDF ",
    fontsize=14,
)
ax.plot(dist2s, ecdf_n, color="blue", alpha=0.8, label="Theoretical CDF")
ax.plot(ecdf.x, ecdf.y, color="#FC5A50", alpha=0.8, label="ECDF for-loop")
ax.plot(ecdf_np.x, ecdf_np.y, color="#08877d", alpha=0.8, label="ECDF Numpy")
ax.set_ylabel("CDF")
ax.set_xlabel("Average Dust per Trial")
ax.set_xlim(130, 410)
plt.legend(loc="best")
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects_sim_hs/output_35_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 4: Using the Normal distribution from Figure 3, as well as the
        empirical distributions found by the simulations, their ECDF functions
        are compared to the CDF of the Normal distribution. The plot shows in
        detail where the three functions differ.
</div>
    


## Research Question 4 - Conclusion

> 4\. Explain how the *Law of Large Numbers* applies here.<br>

By the Law of Large Numbers the sample mean will converge against the population
mean, as the sample size tends to infinity. In a simulation, the number of
samples generated should be as large as possible. This is to ensure that the
empirical mean approximates the population mean as closely as possible.

## Research Question 5 - Conclusion

> 5\. How is the function defined that maps Euro spent to the equivalent dust
value?<br>

For each of the available card pack bundles, (2, 7, 15, 40 packs per bundle
that is) the function mapping Euro spent on card packs to the respective dust
equivalent has different parameter values. The plot shows the line plots for
all available bundles and compares their linear Euro to dust functions. The dust
equivalent, as a ratio of Euro spent to expected dust is lowest for the smallest bundle
containing 2 card packs for 3 Euro and steadily increases up to the
largest bundle of 40 card packs for 50 Euro. Please refer to the plot for more detail.

It is worth noting that the ADV value is less stable for simulations with
fewer card packs per trial and as a result the standard deviation will be
higher for those as well.


```python
bundles = {
    "2": {"qnt": 2, "p": 3},
    "7": {"qnt": 7, "p": 10},
    "15": {"qnt": 15, "p": 20},
    "40": {"qnt": 40, "p": 50},
}
# import matplotlib.pyplot as plt
# import numpy as np
# plt.style.use('science')
# plt.ion()
# plt.close('all')

# mu =254
def dust(bundles=bundles, adv=mu):
    fig, ax = plt.subplots(1,1,figsize=(12,10))
    ax = plt.subplot(111)
    for key in bundles.keys():
        bundles[key]["x_vals"] = [
            x + 1 for x in range(200) if (x + 1) % bundles[key]["p"] == 0
        ]
        bundles[key]["y_vals"] = [((y+1) * mu * bundles[key]['qnt']) for y in
                                  np.arange(len(bundles[key]["x_vals"]))]
        ax.plot(bundles[key]['x_vals'],bundles[key]['y_vals'],label=f"{bundles[key]['qnt']} pack bundle")
    ax.legend(loc="best")
    ax.set_title(f'Euro To Hearthstone Dust Value Plot')
    ax.set_xlabel("Euro Spent On Card Pack Bundles")
    ax.set_ylabel("Dust Equivalent")
    plt.grid(which="major")
    plt.show()

    return bundles

bundlesn = dust()
```


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects_sim_hs/output_37_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 5: The plot shows the linear money spent to equivalent dust value
        functions for the 4 card pack bundles on sale in-game. Not every value
        on the x-axis can be used as input, as only multiples of a single card
        pack bundle are available for purchase. In general, every linear
        combination of the bundles is possible. The most dust per Euro spent
        however is found for multiples of the 50 Euro (<i>40 packs</i>) bundle
        drawn in yellow.
</div>

Thank you very much for reading this article. Please feel free to link to this
article or write a comment in the comments section below.
