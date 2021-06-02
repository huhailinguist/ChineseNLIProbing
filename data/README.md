# 4 Chinese NLI datasets

This folder contains the data files for the 4 Chinese NLI datasets we created for the paper:
_Investigating Transfer Learning in Multilingual Pre-trained Language Models through Chinese Natural Language Inference_.

1. Chinese HANS
2. Chinese stress tests
3. Expanded CLUE diagnostics
4. Chinese semantic fragments

For each dataset, we first describe it in a few sentences, and then present some examples. 

## Purpose

The datasets are intedned to 1) provide more benchmark datasets to test Chinese NLI models, and 
2) create Chinese NLI datasets parallel to existing English data, so as to examine cross-lingual transfer. 

## Summary statistics

|     Dataset                 | category                  | n      |
|----------------------|---------------------------|--------|
| Chinese HANS         | Lexical overlap           | 1,428  |
|                      | Subsequence               | 513    |
| Chinese stress tests         | Distraction: 2 categories | 8,000  |
|                      | Antonym                   | 3,000  |
|                      | Synonym                   | 2,000  |
|                      | Spelling                  | 11,676 |
|                      | Numerical reasoning       | 8,613  |
| Expanded CLUE diagnostics | CLUE original          | 514    |
|                      | CLUE expansion (ours)     | 796    |
|                      | World knowledge (ours)    | 38     |
|                      | Classifier (ours)         | 139    |
|                      | Chengyu/idioms (ours)     | 251    |
|                      | Pro-drop (ours)           | 198    |
|                      | Non-core arguments (ours) | 186    |
| Chinese semantic fragments   | Negation                  | 1,002  |
|                      | Boolean                   | 1,002  |
|                      | Quantifier                | 1,002  |
|                      | Counting                  | 1,002  |
|                      | Conditional               | 1,002  |
|                      | Comparative               | 1,002  |
| sum                  |                           | 43,364 |

## Chinese HANS

### description

Chinese HANS contains adversarial NLI pairs that are aimed to test the models on examples involving one of the two
heuristics: lexical overlap and sub-sequence overlap. This is designed following the English HANS corpus ([McCoy et al 2019](https://github.com/tommccoy1/hans))

中文HANS数据集是为测试OCNLI数据集是否具有偏差(bias)而构建的对抗数据集。测试实例包含两类启发式：(1) 词汇重叠(lexical overlap); (2) 词序列重叠 (subsequence overlap)。数据集的构建基本遵循了英文HANS数据集的构建方法。

| category | premise | hypothesis | label |
|:----:|:----:|:----:|:----:|
| lexical overlap | <font size="4">领导在咖啡馆附近喝啤酒。</font><br/><font size="4">The supervisor drinks beer near the cafe.</font> |<font size="4"> 领导喝啤酒。</font><br/> <font size="4">The supervisor drinks beer.</font>| Entailment |
| lexical overlap | 我们把银行职员留在电影院了。<br/>We left the bank clerk in the cinema. | 银行职员把我们留在电影院了。<br/>The bank clerk left us in the cinema. | Contradiction |
| subsequence overlap | 反正我们吃橘子了。<br/>Anyhow we ate tangerines. | 我们吃橘子了。<br/>We ate tangerines. | Entailment |
| subsequence overlap | 连听昆曲的清洁工都觉得早。<br/>Even the janitors who listen to the Kun opera think it is too early. | 清洁工都觉得早。<br/>Even the janitors think it is too early.| Contradiction |

- datafile: `chinese-hans-full.json`

- statistics: 1428 pairs for lexical overlap; 514 for subsequence

## Chinese stress tests

### description

Chinese stress tests contain 5 large-scale, automatically constructed datasets of adversarial NLI pairs which evaluate systems on the following phenomena:
- distraction: distractions (a tautology or a true statement) are added to either the premise or the hypothesis, which should not influence the inference label.  
- antonym: a word in the premise is replaced with its antonym to form a contradiction. 
- synonym: a word in the premise is replaced with its synonym to form an entailment.
- spelling: a random character in the hypothesis is replaced with its homonym (character with the same *pinyin* ignoring tones).  
- numerical reasoning: a probing set created by extracting premise sentences from a math problem dataset and generating hypotheses following heuristic rules based on quantification.

中文压力测试包括5个大型的、自动生成的对抗数据集， 以测试自然语言推理（NLI）系统能否应对以下现象：
- 干扰： 加入前提句或假设句中的永真式或真实陈述句； 其本身不会影响句子对的标签；
- 反义词：前提句中的一个词被替换为其反义词，形成一个矛盾关系的句子对；
- 同义词：前提句中的一个词被替换为其同义词，形成一个蕴涵关系的句子对；
- 错别字：假设句中的任意一个字符被替换为其同音字（拼音相同，忽略声调）；
- 数值推理：运用与数值相关的启发式改变从数学题数据集中提取的前提句， 形成不同句子对关系的假设句。

#### conditions
For the distraction, antonym, and synonym tests, the data is generated with additional fine-grained conditions to test model abilities of various cues:
- distraction: fine-grained conditions concerning the tautology:
  - added to premise or hypothesis  
  - has negation 
  - the negator (if present)
  - at least one word is not present in the vocabulary of `OCNLI.train`
  - length by character

The result is 100 distinct tautologies that are evenly distributed in our dataset. See the file `distraction_tautology_condition.csv` under `stress_data_final/` for more details.
- antonym: whether the replaced word is an adjective or noun
- synonym: whether the replaced word is an adjective or verb


The design follows the English stress tests ([Naik et al 2018](https://abhilasharavichander.github.io/NLI_StressTest/))). 

### examples
| category | premise | hypothesis | label |
|:----:|:----:|:----:|:----:|
| distraction (add to premise) | <font size="4">国有企业改革的思路和方针政策已经明确, _而且刚做完手术出院的病人不应剧烈运动_。</font><br/><font size="4">The policy of the reform of state-owned enterprises is now clear, _and patients who just had surgery shouldn’t have intense exercise_.</font> |<font size="4"> 根本不存在国有企业。</font><br/> <font size="4">The state-owned enterprises don’t exist.</font>| Contradiction |
| distraction (add to hypothesis) | 这时李家院子挤满了参观的人。<br/>During this time, the Li family’s backyard is full of people who came to visit. | 这地方有个姓李的人家, _而且真的不是假的_。<br/>There is a Li family here, _and true is not false_. | Entailment |
| antonym | 一些地方财政收支矛盾*较大*。<br/>The disagreement about local revenue is relatively _big_. | 一些地方财政收支矛盾*较小*。<br/>The disagreement about local revenue is relatively _small_. | Contradiction |
| synonym | 海部组阁*困难*说明了什么。<br/>What can you tell from the _difficulties_ from Kaifu’s attempt to set up a cabinet? | 海部组阁*艰难*说明了什么。<br/>What can you tell from the _hardships_ from Kaifu’s attempt to set up a cabinet?| Entailment |
| spelling | 身上裹一件工厂发的棉大衣,手插在袖筒里。<br/>(Someone is) wrapped up in a big cotton coat the factory gave with hands in the sleeves. | 身上*质少*一件衣服。<br/>There’s at least \[typo\] one coat on the body.| Entailment |
| numerical reasoning | 小红每分钟打不到510个字。<br/>Xiaohong types fewer than 510 words per min. | 小红每分钟打110个字。<br/>Xiaohong types 110 words per min.| Neutral |

### datafiles
under `stress-data-final/`: `distraction_premise_final.json`; `distraction_hypo_final.json`; `antonym_final.json`; `synonym_final.json`; `spelling_final.json`; `numerical_final.json`;


### statistics 
4000 pairs for distraction (add to premise); 4000 pairs for distraction (add to hypothesis); 3000 pairs for antonym; 2000 pairs for synonym; 2980 pairs for spelling; 8613 pairs for numerical reasoning. 

## Expanded CLUE diagnostics


The CLUE diagnostics dataset contains idiomatic Chinese data written by Chinese linguists, rather than translated Chinese data from English. It originally contains 9 categories including: Lexical Semantics, Comparative, Monotonicity, Argument Structure, Negation, Time of Event, Anaphora, Common Sense, and Double Negation.

We expanded the diagnostic dataset from the Chinese NLU Benchmark (CLUE) (Xu et al., 2020) by:

1) creating diagnostics for the following 5 categories, namely pro-drop, four-character idioms, classifiers, non-core arguments, and world knowledge, among which the first 4 being Chinese-specific linguistic phenomena, and 

2) doubling the number of diagnostic pairs for all 9 existing linguistic phenomena in CLUE with pairs whose premise are selected from a large news corpus (BBC corpus, http://bcc.blcu.edu.cn/) and hypothesis are hand-written by our linguists, to accompany the 514 artificially created data in CLUE.

Examples can be found below.

CLUE诊断数据集包含了中文为母语的语言学者创造的真实语言，而并非从英语来的翻译语。数据集包含了9种语言现象，分别为词类语义、比较、单调性、论元结构、否定、时态语态、回指、常识和双重否定。

在CLUE的基础上，我们从以下两个方面进一步充实了用于实验的数据库：

1. 加入了5类新的语言现象，分别为主语省略、四字成语、量词、非核心论元以及对世界的认知。其中，前四个都是中文所独有的语言现象。

2. 针对CLUE已有的9类语言现象，我们从BBC新闻语料库中挑选了新闻语料作为前提，人工撰写了相对应的假设。

以下是分类例句。

| source | category | premise | hypothesis | label | count |
|:----:|:----:|:----:|:----:|:----:|:----:|
| ours |Idioms               |<font size="4">没权没势的人要来做这份生意，简直就是*白日做梦*。</font><br/><font size="4">For people with no power or background, doing this business is daydreaming.</font> |<font size="4">没权没势的人要做这份生意*非常困难*。</font><br/><font size="4">For people with no power or background, doing this business is very difficult.</font> |Entailment| 251 |
| ours |ProDrop          | <font size="4">吃了三个苹果后，马汉又吃了两个香蕉。</font><br/><font size="4">After eating three apples, Ma Han ate two more bananas.</font>|<font size="4">马汉吃了三个苹果。</font><br/><font size="4">Ma Han ate three apples.</font>|Entailment| 198 |
| ours |NonCoreArguments | <font size="4">生日蛋糕他抹了我一脸。</font><br/><font size="4">He spread the birthday cake all over my face.</font>|<font size="4">他做了生日蛋糕。</font><br/><font size="4">He made a birthday cake.</font> |Neutral | 186 |
| ours |Classifier  | <font size="4">角色集是指一*组*相互依存、相互补充的角色。</font><br/><font size="4">Character set is a group of characters that are dependent on and are supplement of each other.</font>|<font size="4">角色集的角色的个数为*一*。</font><br/><font size="4">The number of characters in a character set is one.</font>|Contradiction| 139 |
| ours |WorldKnowledge  | <font size="4">今天晚上大家在一起吃*年夜饭*。</font><br/><font size="4">People get together to eat the New Year's Eve dinner tonight.</font>|<font size="4">今天是*年三十*。</font><br/><font size="4">Today is the last day of the lunar year.</font> |Entrailment | 38 |
| CLUE |LexicalSemantics| <font size="4">小红很*难过*。</font><br/><font size="4">Xiaohong is  sad.</font></font><font size="4"><br/>小刚经常上课迟到。</font><br/><font size="4">Xiaogang often is late for class.</font>|<font size="4">小红很*难看*。</font><br/><font size="4">Xiaohong is ugly.</font></font><font size="4"><br/>小刚有时上课迟到。</font><br/><font size="4">Xiaogang sometimes is late for class.</font>|Neutral Entailment| 204 |
| CLUE|Comparative     | <font size="4">这筐桔子比那筐*多*。</font><br/><font size="4">This basket has more oranges than that one.</font><br/><font size="4">芒果*比*苹果*好吃*.</font><br/><font size="4">Mangoes taste better than apples.</font>|<font size="4">这筐桔子比那筐*多了不少*。</font><br/><font size="4">This basket has much more oranges than that one.</font><br/><font size="4">芒果*更*好吃.</font><br/><font size="4">Mangoes taste better.</font>|Neutral Entailment| 200 |
| CLUE|Monotonicity    | <font size="4">*有些学生*喜欢在公共澡堂里唱歌。</font><br/><font size="4">Some students like to sing in the shower room.</font><br/><font size="4">她*一大早*就到学校了。</font><br/><font size="4">She arrived at school very early in the morning.</font>|<font size="4">*有些女生*喜欢在公共澡堂里唱歌。</font><br/><font size="4">Some female students like to sing in the shower room.</font></font><br/><font size="4">她到学校了。</font><br/><font size="4">She has arrived at school.</font>| Neutral Entailment | 195 |
| CLUE|ArgumentStructure| <font size="4">小白*看见*小红在打游戏。</font><br/><font size="4">Xiaobai saw Xiaohong playing video games.</font><br/><font size="4">小偷*偷了他一百块钱*。</font><br/><font size="4">A thief stole 100 dollars from him.</font>|<font size="4">小红在打太极拳。</font><br/><font size="4">Xiaohong is doing Tai Chi.</font><br/><font size="4">他*被*偷了一百块钱。</font><br/><font size="4">He got stolen for $100.</font>| Contradiction Entailment | 178 |
| CLUE|Negation     | <font size="4">女生宿舍，男生*勿*入。</font><br/><font size="4">Girls' dormitory, no entering for boys.</font></font><font size="4"><br/>特朗普*没*去上海。</font></font><br/><font size="4">Trump didn't go to Shanghai.</font>|<font size="4">女生宿舍*只能*女生进出。</font><br/><font size="4">Only girls can go in and out of the girls' dormitory.</font><font size="4"><br/>特朗普*没*去北京。</font></font><br/><font size="4">Trump didn't go to Beijing.</font>| Entailment Neutral | 169 |
| CLUE|TimeOfEvent   | <font size="4">记者*去年*采访企业家了。</font><br/><font size="4">The reporter interviewed the entrepreneur last year.</font><br/><font size="4">海峡两岸之间从来没有这么热络*过*!</font><br/><font size="4">The Cross-Strait relations have never been this good!</font>|<font size="4">记者*经常*采访企业家。</font><br/><font size="4">The reporter interviews the entrepreneur very often.</font></font><br/><font size="4">海峡两岸之前比现在热络。</font><br/><font size="4">The Cross-Strait relations were better before!</font>| Neutral Contradiction | 134 |
| CLUE|Anaphora     | <font size="4">马丽和*她*的母亲李琴一起住在这里。</font><br/><font size="4">Ma Li and her mother Li Qin live here together.</font><br/><font size="4">其次，女儿发现老师在课堂上让*她们*画许多东西。</font><br/><font size="4">Second, the daughter found that the teacher let them draw many things in class.</font>|<font size="4">马丽是李琴的母亲。</font><br/><font size="4">Ma Li is Li Qin's mother.</font><br/><font size="4">老师让自己的女儿在课堂上画许多东西。</font><br/><font size="4">The teacher let her own daughter draw many things in class</font>| Contradiction Neutral| 105 |
| CLUE|CommonSense      | <font size="4">小明没有工作。</font><br/><font size="4">Xiaoming doesn't have a job.</font><br/><font size="4">小明现在还是很饿。</font><br/><font size="4">Xiaoming is still hungry.</font>|<font size="4">小明没有住房。</font><br/><font size="4">Xiaoming doesn't have a place to live.</font><br/><font size="4">小明刚刚吃了十个超大汉堡。</font><br/><font size="4">Xiaoming has just eaten 10 giant hamburgers。</font>| Neutral Contradiction| 101 |
| CLUE|DoubleNegation | <font size="4">你*别不*把小病小痛当一回事。</font><br/><font size="4">Don't take minor illness as nothing.</font> <font size="4"><br/>*没有*共产党就*没有*新中国。</font><br/><font size="4">New China won't exist without the Communist Party.</font>|<font size="4"><br/>你应该重视小病小痛。</font><br/><font size="4">You should pay attention to minor illness. </font><font size="4"><br/>国民党也可以建立新中国。</font><br/><font size="4">Kuomintang can also establish a new China.</font>| Entailment Contradiction | 24 |

- datafile: `diagnostics-new-with-categories.json`
- statistics: please refer to the last column of the table above

## Chinese semantic fragments

### description

Following Richardson et al. (2020), we design synthesized fragments to examine models’ understanding ability of six types of linguistic and logic inference: boolean, comparative, conditional, counting, negation and quantifier, where each category has 2-4 templates. 

中文语义片段数据集由自动生成的6类语言学和逻辑学推断对抗数据构成，以检测语言模型对语言学和逻辑推断的学习能力。数据由以下六种语言学和逻辑推断构成：是非，比较，条件，计数，否定和量词，每一种类型包含2到4种生成模板。


| category | premise | hypothesis | label |
|:----:|:----:|:----:|:----:|
| Negation | <font size="4"> *库尔图尔* 只到过 *湛江市麻章区*，*丰隆格* 只到过 *大连市普兰店区*</font> <br/> person<sub>1</sub> only went to location<sub>1</sub>; person<sub>2</sub> only went to location<sub>2</sub> | <font size="4"> *库尔图尔* 没到过 *大连市普兰店区*</font> <br/> person<sub>1</sub> has not been to location<sub>2</sub> | Entailment |
| Boolean |<font size="4"> *何峥、管得宽、李国柱* ... 只到过 *临汾市襄汾县* </font> <br/>person<sub>1</sub>, person<sub>2</sub> ... have only been to location<sub>1</sub> | <font size="4"> *何峥* 没到过 *遵义市红花岗区* </font> <br/> person<sub>1</sub> has not been to location<sub>2</sub> |Entailment |
| Quantifier | <font size="4">有人到过每一个地方，拥抱过每一个人 </font> <br/> Someone has been to every place and hugged every person | <font size="4"> *王艳* 没拥抱过 *包一*  </font> <br/> person<sub>1</sub> hasn't hugged person<sub>2</sub> |Neutral |
| Counting | <font size="4"> *韩声雄* 只拥抱过 *罗冬平、段秀芹 ... 赵常* </font> <br/> person<sub>1</sub> only hugged person<sub>2</sub>, person<sub>3</sub> ... person<sub>8</sub> | <font size="4"> *韩声雄* 拥抱过超过10个人 </font> <br/> person<sub>1</sub> hugged more than 10 people |Contradiction |
| Conditional | <font size="4">  ... *穆肖贝夸* 到过 *赣州市定南县*，如果 *穆肖贝夸* 没到过 *赣州市定南县*，那么 *张本伟* 到过 *呼伦贝尔市阿荣旗*  </font> <br/> ... person<sub>n</sub> has been to location<sub>n</sub>. If personn hasn’t been to location<sub>n</sub>, then person<sub>m</sub> has been to location<sub>m</sub>. | <font size="4"> *张本伟* 没到过 *呼伦贝尔市阿荣旗* </font> <br/> person<sub>m</sub> hasn’t been to location<sub>m</sub> |Neutral |
| Comparative | <font size="4"> *龙银凤* 比 *武书瑾、卢耀辉 ... 奈德哈特* 都小，*龙银凤* 和 *亚厄纳尔普* 一样大 </font> <br/> person<sub>1</sub> is younger than person<sub>2</sub>, ..., person<sub>n</sub>; person<sub>1</sub> is as old as person<sub>m</sub> | <font size="4"> *亚厄纳尔普* 比 *卢耀辉* 大 </font> <br/> person<sub>m</sub> is older than person<sub>2</sub> |Contradiction |

- datafile: 
`new_logic_test.json`
- statistics: 
1000 pairs for each category, 6000 pairs in total.


## References

Below are the references for the four papers where our datasets are based on. 

- English HANS

```
@inproceedings{mccoy-etal-2019-right,
    title = "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference",
    author = "McCoy, Tom  and
      Pavlick, Ellie  and
      Linzen, Tal",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1334",
    doi = "10.18653/v1/P19-1334",
    pages = "3428--3448"
}
```

- English stress tests

```
@inproceedings{naik-etal-2018-stress,
    title = "Stress Test Evaluation for Natural Language Inference",
    author = "Naik, Aakanksha  and
      Ravichander, Abhilasha  and
      Sadeh, Norman  and
      Rose, Carolyn  and
      Neubig, Graham",
    booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
    month = aug,
    year = "2018",
    address = "Santa Fe, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/C18-1198",
    pages = "2340--2353"
}
```

- CLUE diagnostics

```
@inproceedings{xu-etal-2020-clue,
    title = "{CLUE}: A {C}hinese Language Understanding Evaluation Benchmark",
    author = "Xu, Liang  and       Hu, Hai  and       Zhang, Xuanwei  and       Li, Lu  and       Cao, Chenjie  and       Li, Yudong  and       Xu, Yechen  and       Sun, Kai  and       Yu, Dian  and       Yu, Cong  and       Tian, Yin  and       Dong, Qianqian  and       Liu, Weitang  and       Shi, Bo  and       Cui, Yiming  and       Li, Junyi  and       Zeng, Jun  and       Wang, Rongzhao  and       Xie, Weijian  and       Li, Yanting  and       Patterson, Yina  and       Tian, Zuoyu  and       Zhang, Yiwen  and       Zhou, He  and       Liu, Shaoweihua  and       Zhao, Zhe  and       Zhao, Qipeng  and       Yue, Cong  and       Zhang, Xinrui  and       Yang, Zhengliang  and       Richardson, Kyle  and       Lan, Zhenzhong",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.419",
    doi = "10.18653/v1/2020.coling-main.419",
    pages = "4762--4772"
}
```

- English semantic fragments

```
@article{Richardson_Hu_Moss_Sabharwal_2020, 
    title={Probing Natural Language Inference Models through Semantic Fragments}, 
    volume={34}, url={https://ojs.aaai.org/index.php/AAAI/article/view/6397}, 
    DOI={10.1609/aaai.v34i05.6397},
    number={05}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Richardson, Kyle and Hu, Hai and Moss, Lawrence and Sabharwal, Ashish}, 
    year={2020}, 
    month={Apr.}, 
    pages={8713-8721} 
}
```


