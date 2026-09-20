---
layout: post
title: "What I Learned Benchmarking Jev Against Classical Machine Learning"
date: 2026-09-20 12:00:00 +0530
permalink: /blog/jev-vs-classical-ml/
desc: "I tested Jev 1.13.0 against eleven classical classification pipelines on eight datasets. The interesting result was not a universal winner, but a clear map of where each approach worked."
categories: [Machine Learning, Benchmarks]
tags: [Jev, Classical ML, Classification, Benchmarking, NLP, Tabular Data, Balanced Accuracy]
image: /static/portfolio/blog-covers/jev-vs-ml.png
image_alt: "Balanced accuracy results from the Jev and classical machine learning benchmark"
---

I wanted to answer a practical question: if I have a classification problem, how far can a prompt-driven model get before I train a conventional machine-learning pipeline?

The tempting version of this experiment is a race with one winner. That is not what the results showed. Jev was excellent on one sentiment task, competitive on a few other text problems, and much less convincing on the tabular datasets. Classical models remained difficult to beat when structured features and labeled training data were available.

That mixed result is more useful than a sweeping claim. It tells me where Jev may be worth trying first, where a classical baseline remains essential, and which evaluation choices can reverse the apparent winner.

## What I compared

I evaluated Jev 1.13.0 on eight public classification datasets:

- **Text:** AG News, Banking77, SMS Spam, and IMDb.
- **Tabular:** Bank Marketing, Online Shoppers, Breast Cancer, and Iris.

For Jev, I tested two prompting strategies. The zero-shot version received the task and class descriptions but no labeled examples. The few-shot version received one training example per class. Tabular rows were represented as structured feature values rather than converted into free-form prose.

The classical side included logistic regression, SVM, decision tree, random forest, extra trees, k-nearest neighbours, Naive Bayes, histogram gradient boosting, XGBoost, CatBoost, and a voting ensemble. I also kept a majority-class baseline in the report so that every score had some context.

All approaches were evaluated on the same held-out cases for each dataset. Classical models used up to 8,000 training rows, a separate validation set for model selection, and three training seeds. For binary tasks, I also created a second result panel in which decision thresholds were learned from a separate labeled policy split.

The main metric was **balanced accuracy**: the average recall across classes. This matters for datasets such as SMS Spam and Bank Marketing, where plain accuracy can hide poor performance on the minority class.

## The raw results

The table below keeps the comparison readable by showing both Jev variants and the strongest classical mean for each dataset. Values are mean balanced accuracy across three seeds; the best classical model is chosen retrospectively from the eleven pipelines.

| Dataset | Jev zero-shot | Jev few-shot | Best classical result |
|---|---:|---:|---:|
| AG News | 87.5% | 86.3% | 88.4% — SVM |
| Banking77 | 78.9% | 81.9% | 89.7% — SVM |
| SMS Spam | **96.1%** | 95.6% | 95.0% — Naive Bayes |
| IMDb | **96.3%** | 95.9% | 88.4% — Logistic regression |
| Bank Marketing | 53.4% | 55.3% | 71.8% — SVM |
| Online Shoppers | 51.4% | 54.7% | 69.1% — SVM |
| Breast Cancer | 61.0% | 88.8% | 100.0% — SVM / k-NN |
| Iris | 97.0% | 94.5% | 100.0% — multiple models |

The clearest result is IMDb. Zero-shot Jev reached 96.3%, while logistic regression—the strongest classical result in this run—reached 88.4%. A 7.9-point difference is large enough to be interesting even before discussing deployment trade-offs.

Jev also led the raw SMS Spam comparison, although by a much smaller margin. AG News was effectively close: Jev reached 87.5% and SVM reached 88.4%. Banking77 was less favourable to Jev, with SVM ahead by 7.8 points.

The four tabular datasets tell a different story. Jev came close on Iris, but Iris had only 30 test cases and several classical models scored 100%. On the two business-style tabular tasks—Bank Marketing and Online Shoppers—the gap was substantial. Those are the results I would pay attention to before treating a general-purpose model as a drop-in replacement for a trained tabular pipeline.

## IMDb is the standout, not proof of a universal advantage

Sentiment classification is a natural fit for a language model. The label depends on meaning distributed across a review, and the model arrives with a broad prior about language and sentiment. A bag-of-words or similarly bounded classical representation has to learn that relationship from the task data.

That helps explain why IMDb produced Jev's strongest result. It does not prove that Jev is better at text classification in general. Banking77 contains 77 fine-grained intent labels and favoured SVM. AG News was close. The benchmark also did not include fine-tuned transformers, modern embedding pipelines, or other language-model APIs.

My interpretation is narrower: Jev looks especially promising when the task aligns with knowledge already represented by the model and when obtaining task-specific labels is expensive.

## Thresholds changed the SMS result

A classifier has two related jobs. It must separate the classes, and it must choose a decision point. The default decision point is not always the best one, particularly when classes are imbalanced or the cost of false positives and false negatives differs.

I therefore repeated the binary comparisons after selecting thresholds on a separate labeled policy set. That changed the story:

| Dataset | Adjusted Jev zero-shot | Adjusted Jev few-shot | Best adjusted classical result |
|---|---:|---:|---:|
| SMS Spam | 95.9% | 95.8% | **96.3% — Naive Bayes** |
| IMDb | **96.1%** | 95.5% | 88.3% — Logistic regression |
| Bank Marketing | 59.7% | 59.0% | **73.3% — Voting ensemble** |
| Online Shoppers | 51.8% | 53.6% | **71.2% — Voting ensemble** |
| Breast Cancer | 88.4% | 92.7% | **100.0% — Logistic regression** |

The raw SMS lead disappeared: Naive Bayes moved slightly ahead after threshold selection. IMDb barely changed and remained Jev's strongest result. Threshold adjustment helped Jev considerably on Breast Cancer, but not enough to catch the classical models. It also failed to close the large gaps on Bank Marketing and Online Shoppers.

There is an important naming detail here. The adjusted Jev result may use a zero-shot **prompt**, but the complete decision system is no longer zero-shot because labeled policy data was used to select its threshold.

## One example per class was not consistently better

I expected the few-shot prompt to improve most tasks. Instead, examples helped selectively.

The largest gain was Breast Cancer, where raw balanced accuracy rose from 61.0% to 88.8%. Banking77 also improved from 78.9% to 81.9%. But the few-shot mean was lower on AG News, SMS Spam, IMDb, and Iris.

This is a useful warning against treating "add examples" as a universal prompt-engineering recipe. One example per class can clarify an unfamiliar label, but it can also be unrepresentative, introduce accidental wording cues, or narrow the model's interpretation too aggressively. Few-shot selection is part of the model design and should be validated like any other hyperparameter.

## Why I did not calculate one overall winner

It would be easy to average the eight rows and publish a single leaderboard number. I chose not to do that.

The datasets differ in class count, test-set size, modality, difficulty, and practical meaning. A constant prediction gives roughly 1.3% balanced accuracy on the 77-class Banking77 task but 50% on a binary task. Iris contains only 30 test cases; Banking77 contains 1,500. Treating all eight numbers as interchangeable would create an apparently precise result with very little meaning.

I would rather report the pattern:

1. Jev was strongest on IMDb sentiment classification.
2. It was competitive on some other text tasks, but not all of them.
3. Classical pipelines clearly led on the business-oriented tabular datasets.
4. Threshold selection and example choice were large enough to alter the interpretation.

## Limits I would keep in mind

These results are descriptive, not a statistical-significance claim. The displayed variation is across three training seeds, not a confidence interval. Identical successful zero-shot API requests were cached, so the three zero-shot rows are not independent model repetitions.

The classical search was intentionally bounded: four candidates per family, restricted feature budgets, at most 150 trees, and at most 60 histogram-boosting iterations. A larger search or a different representation could improve those models. The comparison also says nothing about the best possible transformer or embedding-based classifier.

The smallest holdouts deserve special caution. Perfect performance on 30 Iris examples or 114 Breast Cancer examples does not imply perfect generalisation. Public-dataset exposure during model pretraining is also unknown.

Finally, Banking77 produced warnings about predictions outside the allowed label set. Failed requests were counted as incorrect, but the published run does not contain enough diagnostics to separate model errors from API failures precisely.

## What I would do on a real project

I would not choose between Jev and classical ML from the model name alone. I would begin with three baselines:

- a simple classical pipeline with careful validation;
- a zero-shot prompt that includes explicit class definitions;
- a small labeled policy set for threshold and error-cost analysis.

For a language task with few labels, Jev may provide a strong starting point and sometimes the best result. For structured business data, I would still expect a tuned classical pipeline to be the default challenger. In both cases, I would inspect per-class errors, latency, cost, stability, and failure handling before thinking about deployment.

The lesson from this benchmark is not that one family replaced the other. It is that pretrained models and classical pipelines have different strengths. The useful engineering decision is knowing which one to test first—and never skipping the baseline that can prove you wrong.

The complete notebook, protocol, and result files are available in the [Jev vs. ML repository](https://github.com/QuicqDev/Jev-vs-ML).
