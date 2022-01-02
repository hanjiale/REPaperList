# REPaperList 

:satisfied: Must-read papers on relation extraction especifically for low-resource setting.



## Datasets
|      Dataset      | #train | #dev | #test | #rel | no_relation | entity type |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |------------- |------------- |
| TACRED [[link](https://nlp.stanford.edu/projects/tacred/) &#124; [paper](https://aclanthology.org/D17-1004)]| 68,124 | 22,631 |15,509 | 42 | &#10004; | &#10004;
| TACREV [[link](https://github.com/DFKI-NLP/tacrev) &#124; [paper](https://aclanthology.org/2020.acl-main.142)]| 68,124 | 22,631 |15,509 | 42 |&#10004; | &#10004;
| Re-TACRED [[link](https://github.com/gstoica27/Re-TACRED) &#124; [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17631)]|  58,465 | 19,584 | 13,418 | 40 |&#10004; | &#10004;
  Wiki80 [[link](https://github.com/thunlp/OpenNRE) &#124; [paper](https://aclanthology.org/D19-3029/)]|  50,400 | 5,600 | -- | 80 |&#10008; | &#10008;
| FewRel 1.0 [[link](https://thunlp.github.io/1/fewrel1.html) &#124; [paper](https://aclanthology.org/D18-1514/)]| 44,800 | 11,200 | 14,000<sup>*</sup>  | 100 (64/16/20) |&#10008; | &#10008;
| FewRel 2.0 [[link](https://thunlp.github.io/2/fewrel2_da.html) &#124; [paper](https://aclanthology.org/D19-1649/)]| 44,800 | 2,500 |  | (64 / 25)  |&#10008; | &#10008;

<font size=2>(*--> unpublic)</font>    
<br>

|      Dataset      | #train | #dev | #test | #rel | #tuples (train &#124; test) | entity overlap type (NEO/EPO/SEO) |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |------------- |------------- |
| NYT24 [[link](https://github.com/nusnlp/PtrNetDecoding4JERE) &#124; [paper](https://aclanthology.org/P18-1047/)]| 56,196 |  |  5,000 | 24  |88,366 &#124; 8,120 | 37,371 / 15,124 / 18,825 &#124; 3,289 / 1,410 / 1,711|
| NYT29 [[link](https://github.com/JiachengLi1995/JointIE) &#124; [paper](https://ojs.aaai.org//index.php/AAAI/article/view/4688)]| 63,306  |  |  4,006 | 29  | 78,973 &#124; 5,859  |  53,444 /  8,379 / 9,862 &#124; 2,963 /  898 / 1,043|
| WebNLG [[link](https://github.com/xiangrongzeng/copy_re) &#124; [paper](https://aclanthology.org/P17-1017/) ] | 5,019 | 500| 703 | 216|||
| ACE05 [[link](https://catalog.ldc.upenn.edu/LDC2006T06)]|||||||
|ACE04 [[link](https://catalog.ldc.upenn.edu/LDC2005T09)]|||||||
|SciERC [[link](http://nlp.cs.washington.edu/sciIE/) &#124; [paper](https://aclanthology.org/D18-1360/) ] | 1,861| 275 |551 |7|||

<br>

## Papers

-----------


### Low-resource relation extraction
#### N-way-K-shot setups
- **FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation.** ![](https://img.shields.io/badge/EMNLP_2018-blue) [[pdf](https://aclanthology.org/D18-1514/)], [[project](https://github.com/thunlp/FewRel)]
- **FewRel 2.0: Towards More Challenging Few-Shot Relation Classification.** ![](https://img.shields.io/badge/EMNLP_2019-blue) [[pdf](https://aclanthology.org/D19-1649/)], [[project](https://github.com/thunlp/fewrel)]
- **Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification.**  ![](https://img.shields.io/badge/AAAI_2019-blue) [[pdf](https://ojs.aaai.org//index.php/AAAI/article/view/4604)], [[project](https://github.com/thunlp/HATT-Proto)]
- **Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification.** ![](https://img.shields.io/badge/ACL_2019-blue) [[pdf](https://aclanthology.org/P19-1277/)], [[project](https://github.com/ZhixiuYe/MLMAN)]
- **Matching the Blanks: Distributional Similarity for Relation Learning.** ![](https://img.shields.io/badge/ACL_2019-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://aclanthology.org/P19-1279/)], [[project](https://paperswithcode.com/paper/matching-the-blanks-distributional-similarity)]
- **Hierarchical Attention Prototypical Networks for Few-Shot Text Classification.** ![](https://img.shields.io/badge/EMNLP_2019-blue) [[pdf](https://aclanthology.org/D19-1045/) ] 
- **Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs.** ![](https://img.shields.io/badge/ICML_2020-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](http://proceedings.mlr.press/v119/qu20a.html)], [[project](https://github.com/DeepGraphLearning/FewShotRE)]
- **Enhance Prototypical Network with Text Descriptions for Few-shot Relation Classification.** ![](https://img.shields.io/badge/CIKM_2020-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://dl.acm.org/doi/10.1145/3340531.3412153)]
- **Learning from Context or Names? An Empirical Study on Neural Relation Extraction.** ![](https://img.shields.io/badge/EMNLP_2020-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://aclanthology.org/2020.emnlp-main.298/)], [[project](https://github.com/thunlp/RE-Context-or-Names)]
- **Bridging Text and Knowledge with Multi-Prototype Embedding for Few-Shot Relational Triple Extraction.** ![](https://img.shields.io/badge/COLING_2020-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://aclanthology.org/2020.coling-main.563/)]
- **Entity Concept-enhanced Few-shot Relation Extraction.** ![](https://img.shields.io/badge/ACL_2021-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://aclanthology.org/2021.acl-short.124/)], [[project](https://github.com/LittleGuoKe/ConceptFERE)]

- **Learning Discriminative and Unbiased Representations for Few-Shot Relation Extraction.** ![](https://img.shields.io/badge/CIKM_2021-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://dl.acm.org/doi/10.1145/3459637.3482268)]]
- **Zero-shot Relation Classification from Side Information.** ![](https://img.shields.io/badge/CIKM_2021-blue) [[pdf](https://dl.acm.org/doi/abs/10.1145/3459637.3482403)], [[project](https://github.com/gjiaying/ZSLRC)]
- **MapRE: An Effective Semantic Mapping Approach for Low-resource Relation Extraction** ![](https://img.shields.io/badge/EMNLP_2021-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://aclanthology.org/2021.emnlp-main.212/)]
- **Exploring Task Difficulty for Few-Shot Relation Extraction.** ![](https://img.shields.io/badge/EMNLP_2021-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://aclanthology.org/2021.emnlp-main.204/)], [[project](https://github.com/hanjiale/hcrp)]
- **Towards Realistic Few-Shot Relation Extraction.** ![](https://img.shields.io/badge/EMNLP_2021-blue) [[pdf](https://aclanthology.org/2021.emnlp-main.433/)], [[project](https://github.com/bloomberg/emnlp21_fewrel)]


#### Generalized few-shot setups
- **KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction.** ![](https://img.shields.io/badge/preprint_2021-blue) ![](https://img.shields.io/badge/BERT-red) [[pdf](https://arxiv.org/abs/2104.07650)], [[project](https://github.com/zjunlp/KnowPrompt)]
- **PTR: Prompt Tuning with Rules for Text Classification.** ![](https://img.shields.io/badge/preprint_2021-blue) ![](https://img.shields.io/badge/RoBERTa-red) [[pdf](https://arxiv.org/abs/2105.11259)], [[project](https://github.com/thunlp/PTR)]
- **GradLRE: Gradient Imitation Reinforcement Learning for Low resource Relation Extraction.** ![](https://img.shields.io/badge/EMNLP_2021-blue) [[pdf](https://aclanthology.org/2021.emnlp-main.216/)], [[project](https://github.com/thu-bpm/gradlre)]
- **Label Verbalization and Entailment for Effective Zero and Few-Shot Relation Extraction.** ![](https://img.shields.io/badge/EMNLP_2021-blue) ![](https://img.shields.io/badge/RoBERTa,DeBERTa-red) [[pdf](https://aclanthology.org/2021.emnlp-main.92/)], [[project](https://github.com/osainz59/Ask2Transformers)]


### Triple Extraction
Joint Extraction of Entities and Relations



### Sentence-level relation extraction
#### Transformer-based models
