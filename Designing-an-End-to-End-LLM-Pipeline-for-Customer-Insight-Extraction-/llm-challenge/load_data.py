"""
Load and prepare data for the extraction pipeline.
Run this after setup.py: python load_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

def verify_api_connection():
    """Verify API connection is working."""
    print("\n" + "="*60)
    print("VERIFYING API CONNECTION")
    print("="*60)
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    base_url = os.getenv('ANTHROPIC_BASE_URL')
    
    if not api_key or not base_url:
        print("❌ API credentials not found in .env")
        return False
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key, base_url=base_url)
        models = client.models.list()
        print(f"✅ API connection successful!")
        print(f"   {len(models.data)} models available")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def load_feedback_data():
    """Load the feedback CSV file."""
    print("\n" + "="*60)
    print("LOADING FEEDBACK DATA")
    print("="*60)
    
    feedback_file = "data/feedback.csv"
    
    if not os.path.exists(feedback_file):
        print(f"❌ File not found: {feedback_file}")
        return None
    
    try:
        df = pd.read_csv(feedback_file)
        print(f"✅ Loaded {feedback_file}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"❌ Error loading feedback data: {e}")
        return None

def load_themes():
    """Load the themes hierarchy."""
    print("\n" + "="*60)
    print("LOADING THEMES HIERARCHY")
    print("="*60)
    
    themes_file = "themes.json"
    
    if not os.path.exists(themes_file):
        print(f"❌ File not found: {themes_file}")
        return None
    
    try:
        with open(themes_file) as f:
            themes = json.load(f)
        
        print(f"✅ Loaded {themes_file}")
        print(f"   Categories: {len(themes['themes'])}")
        
        for category in themes['themes']:
            num_themes = len(category['themes'])
            print(f"     • {category['category']}: {num_themes} themes")
        
        return themes
    except Exception as e:
        print(f"❌ Error loading themes: {e}")
        return None

def analyze_feedback_data(df):
    """Analyze the feedback data."""
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    
    # Calculate metrics
    df['length'] = df['comment'].str.len()
    df['word_count'] = df['comment'].str.split().str.len()
    
    unique_count = df['comment'].nunique()
    duplicate_count = len(df) - unique_count
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total comments: {len(df):,}")
    print(f"   Unique comments: {unique_count:,}")
    print(f"   Duplicates: {duplicate_count:,}")
    
    print(f"\n📏 Comment Length (characters):")
    print(f"   Min:    {df['length'].min()}")
    print(f"   Max:    {df['length'].max()}")
    print(f"   Mean:   {df['length'].mean():.1f}")
    print(f"   Median: {df['length'].median():.1f}")
    print(f"   Std:    {df['length'].std():.1f}")
    
    print(f"\n📝 Word Count:")
    print(f"   Min:    {df['word_count'].min()}")
    print(f"   Max:    {df['word_count'].max()}")
    print(f"   Mean:   {df['word_count'].mean():.1f}")
    print(f"   Median: {df['word_count'].median():.1f}")
    
    # Length distribution
    print(f"\n📌 Length Distribution:")
    length_bins = pd.cut(df['length'], 
                        bins=[0, 50, 200, 500, float('inf')],
                        labels=['0-50', '50-200', '200-500', '500+'])
    
    for bin_label, count in length_bins.value_counts().sort_index().items():
        pct = count / len(df) * 100
        bar = '█' * int(pct / 3)
        print(f"   {bin_label:10s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Sample comments
    print(f"\n📝 Sample Comments:")
    for i, comment in enumerate(df['comment'].head(3), 1):
        length = len(comment)
        words = len(comment.split())
        print(f"\n   {i}. ({length} chars, {words} words)")
        print(f"      {comment[:100]}...")
    
    return df

def save_data_summary(df, themes):
    """Save a summary of loaded data."""
    print("\n" + "="*60)
    print("SAVING DATA SUMMARY")
    print("="*60)
    
    summary = {
        'feedback': {
            'total_comments': len(df),
            'unique_comments': df['comment'].nunique(),
            'duplicates': len(df) - df['comment'].nunique(),
            'length_stats': {
                'min': int(df['length'].min()),
                'max': int(df['length'].max()),
                'mean': float(df['length'].mean()),
                'median': float(df['length'].median())
            },
            'word_count_stats': {
                'min': int(df['word_count'].min()),
                'max': int(df['word_count'].max()),
                'mean': float(df['word_count'].mean()),
                'median': float(df['word_count'].median())
            }
        },
        'themes': {
            'categories': len(themes['themes']),
            'category_list': [cat['category'] for cat in themes['themes']],
            'total_themes': sum(len(cat['themes']) for cat in themes['themes'])
        }
    }
    
    output_file = 'outputs/00_data_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved data summary to {output_file}")
    
    return summary

def main():
    print("""
╔════════════════════════════════════════════════════════╗
║  Loading Data for Extraction Pipeline                  ║
╚════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Verify API
    if not verify_api_connection():
        print("\n❌ Setup not complete. Run setup.py first.")
        return False
    
    # Step 2: Load feedback data
    df = load_feedback_data()
    if df is None:
        return False
    
    # Step 3: Load themes
    themes = load_themes()
    if themes is None:
        return False
    
    # Step 4: Analyze data
    df = analyze_feedback_data(df)
    
    # Step 5: Save summary
    save_data_summary(df, themes)
    
    # Final status
    print("\n" + "="*60)
    print("✅ DATA LOADED AND READY!")
    print("="*60)
    print("\nYou can now run extraction:")
    print("   python src/extract.py")
    print("\nOr explore the data (optional):")
    print("   jupyter notebook notebooks/01_data_exploration.ipynb")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
