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

# Running the Pipeline

## 1. Aspect Extraction

```bash
# Test on 10 comments
python src/extract.py --test

# Run on the full dataset
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

# Full clustering pipeline
python src/cluster.py --full

# Analyze existing results
python src/cluster.py --analyze 
```

## 3. Theme Mapping

```bash
python src/map.py
```

## 4. Evaluation

```bash
python src/evaluate.py
```

outputs/
├── 01_extraction_results.json          # All extracted aspects
├── 01_extraction_summary.json          # Extraction statistics
├── 02_clustering_results.json          # Cluster assignments
├── 02_insights.json                    # Human-readable insights
├── 02_clustering_evaluation.json       # Clustering metrics
├── 03_mappings.json                    # Theme mappings
├── 03_mapping_evaluation.json          # Mapping quality metrics
├── 03_mapping_summary.json             # Quick reference
└── 04_pipeline_evaluation.json         # Overall evaluation
