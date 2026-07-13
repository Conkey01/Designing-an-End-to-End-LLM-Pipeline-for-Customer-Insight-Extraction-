# LLM Challenge - Customer Insight Extraction Pipeline

A complete end-to-end pipeline for extracting, clustering, and mapping customer feedback insights using LLMs.

## Overview

This pipeline takes 5,000 customer feedback comments and produces:
1. **Extracted Aspects** - Granular topics, issues, and features with sentiment
2. **Clustered Insights** - Grouped patterns of similar aspects
3. **Theme Mapping** - Mapped to the provided category/theme hierarchy
4. **Quality Evaluation** - Comprehensive metrics and analysis

---

# Installation

## Prerequisites

- Python 3.8+
- Git

## Quick Setup

```bash
# Clone repository
git clone https://github.com/Conkey01/Designing-an-End-to-End-LLM-Pipeline-for-Customer-Insight-Extraction-.git
cd Designing-an-End-to-End-LLM-Pipeline-for-Customer-Insight-Extraction-/llm-challenge

# Install dependencies
python setup.py

# Enter your API key when prompted

# Download and prepare the dataset
python load_data.py
```

---
## List of all files

Initial files:

From the llm-challenge:

themes.json

data/feedback.csv

README.md 

Files I created:

For analysis (always good practice to understand, plot and calculate simple metrics of the data at each stage of the pipeline):

notebooks/01_data_exploration.ipynb - analysis of the original comment data 

notebooks/02_extraction_analysis.ipynb - analysis of the data after the extraction step

notebooks/03_clustering_analysis.ipynb - analysis of the data after the clustering step


README.md - this readme file 

requirements.txt - list of all depenedencies and versions needed 

setup.py - run the set up code

load_data.py - download and prepare the dataset


src folder contains all python scripts to run steps 1-3 and the evaluations at each stage:

src/extract.py - to do the extraction step

src/cluster.py - to do the cluster step

src/evaluate_extraction.py - to evaluate the extraction step

src/evaluate_clustering.py - to evaluate the clustering step

src/map.py - to do the mapping step

src/evaluate_mapping.py - to evaluate the mapping step


outputs:

outputs/00_data_summary.json - summary stats of the original data

outputs/01_length_distribution.png - analysis of the original data

outputs/02_sentiment_distribution.png - analysis of the original data

outputs/03_top_terms_tfidf.png - analysis of the original data

outputs/phase0_sample_20comments.csv - sample of 20 random comments

outputs/01_extraction_results.json - results of the extraction step

outputs/01_extraction_results.jsonl - results of the extraction step

outputs/02_clustering_evaluation.json - evaluation results of the clustering step

outputs/02_clustering_results.json - results of the clustering step

outputs/02_insights.json - summary of all the clusters from the clustering step

outputs/cluster_sentiment_composition.png - analysis of the data after clustering

outputs/cluster_sizes_bar_chart.png - analysis of the data after clustering

outputs/01_extraction_evaluation.json - evaluation results after the extraction step

outputs/02_clustering_evaluation_detailed.json - evaluation results of the clustering step

outputs/03_confidence_report.json - mappings confidence scores 

outputs/03_theme_distribution.json - results of the mappings step

outputs/03_mapping_results.json - results of the mappings step

# Running the Pipeline

## 1. Aspect Extraction

```bash
# Test on 10 comments
python src/extract.py --test

# Run on the full dataset - if any run this one for the full pipeline
python src/extract.py --full

# Run both the test and full pipeline
python src/extract.py
```

## 2. Clustering

```bash
# Test on 10-20% of data
python src/cluster.py --test

# Test with visualization
python src/cluster.py --test --viz

# Full clustering pipeline - if any run this one for the full pipeline
python src/cluster.py --full

# Analyze existing results
python src/cluster.py --analyze 
```

## 3. Theme Mapping

```bash
# Semantic-only mapping
python src/map.py --semantic

# Semantic + selective LLM validation
python src/map.py --validate

# Full LLM-based mapping - if any run this for the full pipeline
python src/map.py --full

# Evaluate existing mapping results
python src/map.py --evaluate 
```

## 4. Evaluation

```bash
# Evaluation for the extraction step
python src/evaluate_extraction.py

# Evaluation for the clustering step
python src/evaluate_clustering.py

# Evaluation for the mapping step
python src/evaluate_mapping.py
```

# Methods/Evaluation

## 1. Extraction

## Methods

We input raw customer comments and send it to Claude Haiku with the below structured prompt:
"You are an expert at extracting structured insights from customer feedback.

TASK: Extract specific aspects (topics, issues, features) mentioned in this customer comment along with the sentiment expressed towards each aspect.

IMPORTANT GUIDELINES:
1. Aspects should be GRANULAR and SPECIFIC (not high-level categories like "app" or "service")
   - GOOD: "App login speed", "Sign-up process complexity", "Push notification frequency"
   - BAD: "App experience", "Service", "Features"

2. Extract aspects that are EXPLICITLY or CLEARLY IMPLIED in the text
   - Do not invent aspects not mentioned
   - Include implicit aspects if they're reasonably inferred from context

3. Sentiment should be: "positive", "negative", or "neutral"
   - Evaluate sentiment TOWARDS THAT SPECIFIC ASPECT, not overall
   - Handle mixed sentiments per aspect if needed

4. Include an evidence snippet from the comment supporting each aspect

5. Confidence score should reflect how certain you are about this extraction (0.0-1.0)

OUTPUT FORMAT (JSON only, no other text):
{{
  "aspects": [
    {{
      "aspect": "<specific aspect name>",
      "sentiment": "<positive|negative|neutral>",
      "evidence": "<relevant quote from comment>",
      "confidence": <0.0-1.0>
    }}
  ]
}}

COMMENT TO ANALYZE:
{comment}

Respond ONLY with valid JSON."

## Evaluation

This is supposed to extract aspects, sentiment and confidence. The output is JSON with structured aspects. 
This gave good extractions for a low cost rather than using human annotation or finetuning a large model which could cost more than the 20 dollar API budget. It's also better than using a heuristic/Regex/rule-based approach which is not scalable/fast enough for our needs. Claude Haiku has the right balance between cost and model performance compared to GPT-4/open source LLMs for example.
The model scaled well and was relatively fast in processing the 5k comments, no need for finetuning and we used batched processing to handle the cost and size of the data. 
We have a JSON output format that we can parse reliably.

We measure good extraction firstly through the success rate of how many comments we were able to extract information from in total. If we are only able to extract information from a low percentage of them then we have wasted our API budget. We also get around this by first testing the extraction step on a small subset of the comments to make sure the pipeline is working before we waste our budget on a pipeline that does not work.

A second way to measure good extraction is the confidence scores we prompt the LLM to give us for each aspect which tell us which aspects to trust. If needed we could filter out low confidence aspects. A low confidence score <0.5 can tell us which are unreliable whereas a high score ~1 can tell us which are high quality. We can aggregate a comments confidence scores by taking the mean/median of each aspects confidence score associated with that comment. If 80% of aspects are of a high quality say >0.8 then we can say around 8 in 10 aspects are of a high quality. 

We can also measure the number of aspects per comment. We can take the average aspects per comment, between around 3-5 gives good granularity, whereas a high average could suggest we are over extracting and we are picking up to much noise. A low average could suggest we are missing too much detail and we have very sparse comments. 

We can also look at the sentiment distribution, looking at the percentage of aspects that are positive, negative or neutral. If they are too heavy tailed to one of them then they could be biased, too many positive reviews could suggest they are fake/filtered reviews or too many negatives then they could be from a complaints forum. We would expect a natural mix of sentiments and might be suspicious if they are heavy swayed to one class.

We can also measure how many aspects are unique/duplicates. If there is a lot of duplicates then themes are consistent and it is a good sign that clustering will work well. A lot of unique aspects could show that the model is picking up a lot of noise and there is many patterns in the data which could hinder clustering in the next step. 

We can also track whether similar comments give similar/the same aspects to see whether the extraction remains consistent between comments which again can show that clustering would work well as we would expect similar aspects to group together. 

Finally we can also track the false positive rate which we want to be low as these would introduce noise and won't help clustering. For example if our comment says I am happy with the service then a false positive would be the extraction saying issues with the service and we don't want that.

## Results

Run the following to get the evaluation metrics mentioned:

```bash
python src/evaluate_extraction.py
```

It will save the results to the outputs/01_extraction_evaluation.json file 

Here is a small summary of the results for this section:

Key Strengths:

   98.5% success rate (very reliable)
   
   0.847 mean confidence (high quality aspects)
   
   4.07 aspects per comment (good granularity)
   
   55% negative, 31% positive (realistic distribution)
   
   52% redundancy (strong patterns)
   
   \$0.0003 per aspect (excellent value)

   The extraction results are saved to the outputs/01_extraction_results.json file 

   
## 2. Clustering

## Methods 

We input the aspects we extracted from the previous step and generate emeddings using sentence-transformers. Effectively converting each aspect to a 384-dimensional vector. We find optimal number of clusters for K-means clustering by running the algorithm for every possible k between 3 to 50 and picking the k with the highest silhouette score which we get as k=38. Each aspect is then assigned a cluster 0-37 and then we can generate insights for each cluster.

## Evaluation

For clustering we can evaluate the specific algorithm we have chosen which was k-means. While k-means is free, very fast and very interpretable, the silhouette score we got was very poor even for the best model. Given the data we have I think now an algorithm that has soft clustering is a better choice. K-means does not handle overlapping clusters very well and I think with the aspects we have some of them are very similar/practically the same thing especially when we have so many clusters some of them are very closely linked and will have some overlap. So I think to improve the pipeline and improve the clusters we should use a soft clustering method (like GMMs etc.) instead that gives probabilities that each comment belongs to a cluster rather than it just being fixed on one cluster because aspects can be part of multiple clusters in the task and with the data we have. So with more time I would change the clustering method. 

The silhouette score can tell us how well seperated the clusters are whereas the Davies-Bouldin Score can tell us how compact the clusters are and we use this to measure performance too.

We can also look at the cluster distributions and see how many instances are in each cluster, how many are in the largest and smallest clusters and hope that the range between these is not too large and that we have well balanced clusters. We want aspects to be distributed meaningfully and not lots of them being cluster into the same clusters leaving very small and very large cluster shapes.

We can also make sure the clusters actually make sense, do clustered aspects actually belong together? (are they all about the same thing?). Do they all have a clear theme and are not randomly grouped?

Do clusters have dominant sentiments? If clusters are mainly of positive or negative aspects instead of the other then that can be a good sign that the clusters have grouped together well.

We can also look at the percentage of aspects that were clustered successfully.

Finally something I have not done but could be done is to actually measure how good the embeddings are aswell do these capture the patterns and semantic relationships in the data well. We could do this by using dimensionality reduction techniques (like PCA, t-SNE etc.) and visualise whether similar aspects cluster together in the embeddings spaces. We could visualise our clusters in these spaces to see how well we have clustered the data and to see the overlap of clusters we think should be there a bit more.


## Results

Run the following to get the evaluation metrics mentioned:

```bash
python src/evaluate_clustering.py
```

It will save the results to the outputs/02_clustering_evaluation_detailed.json file 

Here is a small summary of the results for this section:

Key Strengths:

  Silhouette 0.0657 (FAIR, expected for customer feedback with overlapping topics)
  
  Davies-Bouldin 3.16 (FAIR, consistent with silhouette)
  
  Cluster size ratio 7.4x (GOOD, well-balanced)
  
  Mean sentiment coherence 75%+ (GOOD, clear dominant sentiments)
  
  100% coverage (EXCELLENT, all aspects clustered)
  
  100% interpretability (EXCELLENT, all clusters have names)
  
  Fast computation (EXCELLENT, 2.5 minutes total)
  
  Low cost (EXCELLENT, \$0.07 for insights)

## 3. Mapping

## Methods 
For this task we can make a choice between using LLMs or semantic similarity or both (dependent on the terminal command we run from the pipeline). If we use both we first create embeddings for all 15 themes using sentence-transformers. Then for each cluster (we can embed the cluster summary to get semantic understanding) we compute semantic similarity (cosine similarity) to all themes and then return the top-3 matching themes with confidence scores (Claude can express uncertainty). Finally we Use Claude to select primary theme with reasoning, given a structured prompt similar to that in the extraction step but obviously for this task instead. We can alterntively just restrict ourselves to only using LLMs or only using semantic similarity instead aswell. 

## Evaluation
A combination of semantic similarity and LLMs is fast, interpretable and cost efficient compared to finrtuning a large model/human annotation or a rule based/heuristic approach. The cost is now low because we are dealing with small numbers of clusters aswell rather than 5k comments 


## Results

Run the following to get the evaluation metrics mentioned:

```bash
python src/evaluate_mapping.py
```

It will save the results to the outputs/03_mapping_evaluation_detailed.json file 

Here is a small summary of the results for this section:

Key Strengths:

  Mean confidence 0.889 (EXCELLENT - very sure mappings)
  
  All 38/38 mappings >0.75 confidence (no uncertain mappings)
  
  10/15 themes used (good coverage, not forcing unused)
  
  All 5 categories represented (excellent breadth)
  
  Balanced category distribution (no concentration)
  
  70% of themes map multiple clusters (consistent patterns)
  
  All sample mappings make business sense (valid)
  
  Only \$0.03 per cluster (exceptional value)
  
  5.5% of budget used (plenty remaining)
