# [When Stereotypes GTG: The Impact of Predictive Text Suggestions on Gender Bias in Human-AI Co-Writing](https://arxiv.org/abs/2409.20390)

By: [Connor Baumler](https://ctbaumler.github.io/) `<baumler@cs.umd.edu>` and Hal Daumé III

```
@inproceedings{10.1145/3772318.3790733,
  author = {Baumler, Connor and Daum\'{e}, Hal, III},
  title = {When Stereotypes GTG: The Impact of Predictive Text Suggestions on Gender Bias in Human-AI Co-Writing},
  year = {2026},
  isbn = {9798400722783},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3772318.3790733},
  doi = {10.1145/3772318.3790733},
  booktitle = {Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems},
  articleno = {682},
  numpages = {44},
  keywords = {Co-writing, predictive text, stereotyping},
  location = {
  },
  series = {CHI '26}
}
```

This repository contains writing scenarios, model prompts, and sample generation code. The human-facing prefix of each writing scenario and the corresponding model prompt (and potential pre-determined first set of suggestions) can be found in ``prompts.tsv``. The prompts used to collect model annotations can be found in ``hypotheses.csv``. ``example_generation.py`` includes the code for generating predictive text suggestions and can be run as-is to see example generations on one scenario.
