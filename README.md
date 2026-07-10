# LLM Challenge - Customer Insight Extraction Pipeline

A complete end-to-end pipeline for extracting, clustering, and mapping customer feedback insights using LLMs.

## Overview

This pipeline takes 5,000 customer feedback comments and produces:
1. **Extracted Aspects** - Granular topics, issues, and features with sentiment
2. **Clustered Insights** - Grouped patterns of similar aspects
3. **Theme Mapping** - Mapped to the provided category/theme hierarchy
4. **Quality Evaluation** - Comprehensive metrics and analysis

## Quick Start

### Prerequisites
- Python 3.8+
- API key for Anthropic-compatible endpoint (provided separately)

### Installation

```bash
# Clone repository
git clone https://github.com/chattermill/llm-challenge.git
cd llm-challenge

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export ANTHROPIC_API_KEY="sk-..."
export ANTHROPIC_BASE_URL="https://llm-api.datascience.chattermill.xyz/anthropic"
