# Chinese NLI Probing

## Introduction

This folder contains data and code for the paper [**Investigating Transfer Learning in Multilingual Pre-trained Language Models through Chinese Natural Language Inference**](https://arxiv.org/abs/2106.03983), where we examine the cross-lingual transfer learning in [XLM-R](https://arxiv.org/abs/1911.02116) via 4 categories of newly constructed NLI datasets:

- Chinese HANS
- Chinese stress tests
- Expanded CLUE diagnostics
- Chinese semantic fragments

All of them are made in parallel to existing adversarial/probing datasets in English (but from scratch, i.e., no translations),
so as to allow cross-lingual transfer studies in both directions: English-to-Chinese and Chinese-to-English. 


## Data

The generated data can be found in the `data` folder, with detailed explanations. 

## Code

The code used for data generation can be found in the `code` folder (will be uploaded shortly).

## Main results

We  find  thatcross-lingual  models  trained  on  English  NLI do transfer well across our Chinese tasks (e.g., in  3/4  of  our  challenge  categories,  they  per-form as well/better than the best monolingual models,  even  on  3/5  uniquely  Chinese  lin-guistic phenomena such as _idioms_, _pro drop_). These results,  however,  come with importantcaveats:   cross-lingual  models  often  performbest when trained on a mixture of English and high-quality monolingual NLI data (OCNLI), and are often hindered by automatically trans-lated resources (XNLI-zh). For many phenomena, all models continue to struggle, highlighting the need for our new diagnostics to help benchmark Chinese and cross-lingual models.

(will be expanded shortly)


## References

If you use our resources or find our results useful, please cite:

```
@inproceedings{hu-et-al-2021-investigating,
	title={Investigating Transfer Learning in Multilingual Pre-trained Language Models through {Chinese} Natural Language Inference},
	author={Hai Hu and He Zhou and Zuoyu Tian and Yiwen Zhang and Yina Ma and Yanting Li and Yixin Nie and Kyle Richardson},
	booktitle={Findings of ACL},
	year={2021}
}
```

