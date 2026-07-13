"""
Stage 2: Create Insights - Cluster extracted aspects into coherent patterns.

Usage:
    python src/cluster.py --test          # Test on 10-20% of data
    python src/cluster.py --test --viz    # Test with visualization
    python src/cluster.py --full          # Full clustering pipeline
    python src/cluster.py --analyze       # Analyze existing results
"""

import json
import os
import sys
import argparse
import logging
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic

# Clustering libraries
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False

from tqdm import tqdm

# Load environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL = "claude-haiku-4-5"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

EXTRACTION_RESULTS = OUTPUT_DIR / "01_extraction_results.json"
CLUSTERING_RESULTS = OUTPUT_DIR / "02_clustering_results.json"
INSIGHTS_FILE = OUTPUT_DIR / "02_insights.json"
EVALUATION_FILE = OUTPUT_DIR / "02_clustering_evaluation.json"
COST_TRACKER = OUTPUT_DIR / ".cost_tracker.json"

# Budget tracking
BUDGET_LIMIT = 20.0
HAIKU_INPUT_COST = 0.80 / 1_000_000
HAIKU_OUTPUT_COST = 4.0 / 1_000_000


class AspectProcessor:
    """Process and normalize extracted aspects."""
    
    @staticmethod
    def load_extraction_results() -> Tuple[List[Dict], int]:
        """Load results from Stage 1."""
        
        if not EXTRACTION_RESULTS.exists():
            raise FileNotFoundError(f"Stage 1 results not found: {EXTRACTION_RESULTS}")
        
        with open(EXTRACTION_RESULTS) as f:
            data = json.load(f)
        
        # Extract successful results
        aspects_data = []
        for result in data['results']:
            if result['status'] == 'success':
                for aspect in result['extraction']['aspects']:
                    aspects_data.append({
                        'aspect': aspect['aspect'],
                        'sentiment': aspect['sentiment'],
                        'evidence': aspect.get('evidence', ''),
                        'confidence': aspect.get('confidence', 0.5),
                        'source_comment': result['comment']
                    })
        
        logger.info(f"✅ Loaded {len(aspects_data)} aspects from Stage 1")
        total_comments = data['metadata']['successful']
        
        return aspects_data, total_comments
    
    @staticmethod
    def deduplicate_aspects(aspects: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate aspects (same aspect mentioned multiple times).
        Keeps the one with highest confidence.
        """
        
        aspect_dict = {}
        for aspect_obj in aspects:
            aspect = aspect_obj['aspect'].lower().strip()
            
            if aspect not in aspect_dict:
                aspect_dict[aspect] = aspect_obj
            else:
                # Keep the one with higher confidence
                if aspect_obj['confidence'] > aspect_dict[aspect]['confidence']:
                    aspect_dict[aspect] = aspect_obj
        
        deduped = list(aspect_dict.values())
        logger.info(f"Deduplicated: {len(aspects)} → {len(deduped)} unique aspects")
        
        return deduped
    
    @staticmethod
    def normalize_aspect_names(aspects: List[Dict]) -> List[Dict]:
        """Normalize aspect names for consistency."""
        
        for aspect_obj in aspects:
            # Lowercase, strip whitespace
            aspect_obj['aspect'] = aspect_obj['aspect'].strip().lower()
            # Remove common redundancies
            aspect_obj['aspect'] = aspect_obj['aspect'].replace('the ', '')
        
        return aspects


class AspectEmbedder:
    """Generate embeddings for aspects using sentence transformers or LLM."""
    
    def __init__(self):
        """Initialize embedder."""
        # Try to use sentence-transformers (lightweight, no API calls)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.method = 'sentence-transformers'
            logger.info("✅ Using sentence-transformers for embeddings")
        except ImportError:
            logger.warning("⚠️  sentence-transformers not available")
            logger.info("   Installing: pip install sentence-transformers")
            self.model = None
            self.method = 'none'
    
    def embed_aspects(self, aspects: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for aspects.
        
        Args:
            aspects: List of aspect dictionaries
            
        Returns:
            Embeddings matrix (N, embedding_dim)
        """
        
        if self.method == 'none':
            logger.error("Embedder not initialized. Install sentence-transformers.")
            raise RuntimeError("No embedding method available")
        
        aspect_texts = [a['aspect'] for a in aspects]
        
        logger.info(f"Generating embeddings for {len(aspect_texts)} aspects...")
        embeddings = self.model.encode(aspect_texts, show_progress_bar=True)
        
        logger.info(f"✅ Generated embeddings: shape {embeddings.shape}")
        
        return embeddings


class ClusteringEngine:
    """Cluster aspects into coherent groups."""
    
    def __init__(self):
        """Initialize clustering engine."""
        self.embedder = AspectEmbedder()
    
    def find_optimal_clusters(self, embeddings: np.ndarray, 
                             aspects: List[Dict],
                             max_clusters: int = 50) -> Tuple[np.ndarray, int]:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            embeddings: Aspect embeddings
            aspects: Aspect list (for context)
            max_clusters: Max clusters to test
            
        Returns:
            Cluster labels and optimal k
        """
        
        # Limit search based on number of aspects
        max_k = min(max_clusters, len(aspects) // 3)  # At least 3 aspects per cluster
        max_k = max(3, max_k)  # At least 3 clusters
        
        logger.info(f"Finding optimal clusters (testing k=3 to {max_k})...")
        
        silhouette_scores = []
        cluster_range = range(3, max_k + 1)
        
        for k in tqdm(cluster_range, desc="Testing cluster counts"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
        
        # Find optimal k
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(cluster_range)[optimal_idx]
        optimal_score = silhouette_scores[optimal_idx]
        
        logger.info(f"✅ Optimal k={optimal_k} (silhouette score: {optimal_score:.4f})")
        
        # Fit with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        return labels, optimal_k
    
    def cluster_aspects(self, embeddings: np.ndarray,
                       aspects: List[Dict],
                       method: str = 'kmeans') -> Tuple[np.ndarray, Dict]:
        """
        Cluster aspects using specified method.
        
        Args:
            embeddings: Aspect embeddings
            aspects: Aspect list
            method: 'kmeans', 'hdbscan', or 'hierarchical'
            
        Returns:
            Cluster labels and metadata
        """
        
        logger.info(f"\n{'='*70}")
        logger.info(f"CLUSTERING: {method.upper()}")
        logger.info(f"{'='*70}")
        
        logger.info(f"Clustering {len(aspects)} aspects using {method}...")
        
        if method == 'kmeans':
            labels, optimal_k = self.find_optimal_clusters(embeddings, aspects)
            metadata = {'method': 'kmeans', 'n_clusters': int(optimal_k)}
            
        elif method == 'hdbscan':
            clusterer = HDBSCAN(min_cluster_size=3)
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            metadata = {'method': 'hdbscan', 'n_clusters': int(n_clusters), 'noise_points': int(sum(labels == -1))}
            
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=20)  # Conservative estimate
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels))
            metadata = {'method': 'hierarchical', 'n_clusters': int(n_clusters)}
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate metrics
        if len(set(labels)) > 1 and -1 not in labels:
            silhouette = silhouette_score(embeddings, labels)
            davies_bouldin = davies_bouldin_score(embeddings, labels)
            metadata['silhouette_score'] = float(silhouette)
            metadata['davies_bouldin_score'] = float(davies_bouldin)
            
            logger.info(f"   Silhouette score: {silhouette:.4f}")
            logger.info(f"   Davies-Bouldin score: {davies_bouldin:.4f}")
        
        return labels, metadata
    
    @staticmethod
    def organize_clusters(aspects: List[Dict], 
                         labels: np.ndarray) -> Dict[int, List[Dict]]:
        """
        Organize aspects into clusters.
        
        Args:
            aspects: List of aspects
            labels: Cluster labels
            
        Returns:
            Dict mapping cluster_id -> list of aspects
        """
        
        clusters = {}
        for aspect, label in zip(aspects, labels):
            label_int = int(label)  # Convert to Python int
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(aspect)
        
        return clusters


class InsightGenerator:
    """Generate natural language insight descriptions for clusters."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize Claude client."""
        self.client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL")
        )
        self.input_tokens = 0
        self.output_tokens = 0
    
    def generate_insight_description(self, cluster_aspects: List[Dict], 
                                    cluster_id: int) -> Dict:
        """
        Generate a natural language description of a cluster.
        
        Args:
            cluster_aspects: Aspects in this cluster
            cluster_id: Cluster identifier
            
        Returns:
            Insight description dict
        """
        
        # Format aspects
        aspects_text = "\n".join([
            f"- {a['aspect']} ({a['sentiment']}, confidence: {a['confidence']:.2f})"
            for a in cluster_aspects
        ])
        
        prompt = f"""You are an expert at synthesizing customer feedback patterns into actionable insights.

TASK: Analyze this cluster of customer feedback aspects and generate a concise, clear insight description that captures the underlying pattern or theme.

CLUSTER ASPECTS:
{aspects_text}

REQUIREMENTS:
1. Give the cluster a SHORT, DESCRIPTIVE NAME (2-4 words max)
2. Write a 1-2 sentence summary explaining what this cluster represents
3. Identify the PRIMARY SENTIMENT (mostly positive, mostly negative, mixed)
4. List 2-3 KEY THEMES or patterns you notice
5. Suggest potential business impact (if relevant)

OUTPUT FORMAT (JSON only):
{{
  "cluster_name": "<short name>",
  "summary": "<1-2 sentence summary>",
  "primary_sentiment": "<positive|negative|mixed>",
  "key_themes": ["<theme 1>", "<theme 2>", "<theme 3>"],
  "business_impact": "<brief impact statement or null>"
}}

Generate ONLY valid JSON."""
        
        try:
            message = self.client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.input_tokens += message.usage.input_tokens
            self.output_tokens += message.usage.output_tokens
            
            response_text = message.content[0].text.strip()
            
            # Parse JSON (handle markdown wrapping)
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            insight = json.loads(response_text)
            
            return {
                'cluster_id': int(cluster_id),
                'insight': insight,
                'num_aspects': len(cluster_aspects),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error generating insight for cluster {cluster_id}: {e}")
            return {
                'cluster_id': int(cluster_id),
                'status': 'error',
                'error': str(e)
            }
    
    def generate_insights_for_all_clusters(self, clusters: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Generate insights for all clusters.
        
        Args:
            clusters: Dict of cluster_id -> aspects
            
        Returns:
            List of insight descriptions
        """
        
        insights = []
        
        logger.info(f"\nGenerating insights for {len(clusters)} clusters...")
        
        for cluster_id in tqdm(sorted(clusters.keys()), desc="Generating insights"):
            cluster_aspects = clusters[cluster_id]
            insight = self.generate_insight_description(cluster_aspects, cluster_id)
            insights.append(insight)
        
        return insights
    
    def get_cost_summary(self) -> Dict:
        """Calculate API costs."""
        input_cost = self.input_tokens * HAIKU_INPUT_COST
        output_cost = self.output_tokens * HAIKU_OUTPUT_COST
        total_cost = input_cost + output_cost
        
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.input_tokens + self.output_tokens,
            'input_cost_usd': round(input_cost, 4),
            'output_cost_usd': round(output_cost, 4),
            'total_cost_usd': round(total_cost, 4)
        }


class ClusteringEvaluator:
    """Evaluate clustering quality."""
    
    @staticmethod
    def calculate_cluster_statistics(clusters: Dict[int, List[Dict]],
                                    labels: np.ndarray) -> Dict:
        """
        Calculate statistics about clusters.
        
        Args:
            clusters: Cluster dict
            labels: Cluster labels array
            
        Returns:
            Statistics dict
        """
        
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        
        return {
            'num_clusters': int(len(clusters)),
            'cluster_sizes': {
                'min': int(min(cluster_sizes)),
                'max': int(max(cluster_sizes)),
                'mean': float(np.mean(cluster_sizes)),
                'median': float(np.median(cluster_sizes))
            },
            'cluster_size_distribution': {
                str(k): int(v) for k, v in zip(
                    sorted(clusters.keys()),
                    sorted(cluster_sizes)
                )
            }
        }
    
    @staticmethod
    def analyze_sentiment_distribution(clusters: Dict[int, List[Dict]]) -> Dict:
        """
        Analyze sentiment distribution across clusters.
        
        Args:
            clusters: Cluster dict
            
        Returns:
            Sentiment analysis dict (with string keys for JSON serialization)
        """
        
        analysis = {}
        
        for cluster_id, aspects in clusters.items():
            sentiments = {}
            for aspect in aspects:
                sentiment = aspect['sentiment']
                sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
            
            # Convert cluster_id key to string for JSON serialization
            analysis[str(cluster_id)] = sentiments
        
        return analysis
    
    @staticmethod
    def generate_evaluation_report(clusters: Dict[int, List[Dict]],
                                  labels: np.ndarray,
                                  embeddings: np.ndarray,
                                  insights: List[Dict],
                                  metadata: Dict) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            clusters: Cluster dict
            labels: Cluster labels
            embeddings: Aspect embeddings
            insights: Generated insights
            metadata: Clustering metadata
            
        Returns:
            Evaluation report dict
        """
        
        stats = ClusteringEvaluator.calculate_cluster_statistics(clusters, labels)
        sentiment_analysis = ClusteringEvaluator.analyze_sentiment_distribution(clusters)
        
        successful_insights = len([i for i in insights if i['status'] == 'success'])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'clustering_method': metadata.get('method', 'unknown'),
            'clustering_metrics': metadata,
            'cluster_statistics': stats,
            'sentiment_distribution': sentiment_analysis,
            'insights_generation': {
                'total_clusters': int(len(clusters)),
                'successful_insights': int(successful_insights),
                'failed_insights': int(len(clusters) - successful_insights),
                'success_rate': float(successful_insights / len(clusters)) if clusters else 0.0
            },
            'quality_assessment': {
                'coverage': float(len([i for i in insights if i['status'] == 'success']) / len(clusters) * 100) if clusters else 0.0,
                'coherence': "High" if metadata.get('silhouette_score', 0) > 0.3 else "Medium" if metadata.get('silhouette_score', 0) > 0.1 else "Low",
                'note': "Silhouette score > 0.5 indicates strong clustering, 0.3-0.5 is moderate, < 0.3 is weak"
            }
        }
        
        return report


def run_test_clustering():
    """Test clustering on a sample of data."""
    
    print("\n" + "="*70)
    print("STAGE 2: TEST CLUSTERING (SAMPLE DATA)")
    print("="*70)
    
    # Load and prepare data
    aspects, total_comments = AspectProcessor.load_extraction_results()
    
    # Sample ~10-20% for testing
    sample_size = max(50, len(aspects) // 5)
    sample_indices = np.random.choice(len(aspects), sample_size, replace=False)
    sample_aspects = [aspects[i] for i in sample_indices]
    
    logger.info(f"\nSampling {len(sample_aspects)} aspects ({len(sample_aspects)/len(aspects)*100:.1f}%) for testing")
    
    # Deduplicate and normalize
    sample_aspects = AspectProcessor.deduplicate_aspects(sample_aspects)
    sample_aspects = AspectProcessor.normalize_aspect_names(sample_aspects)
    
    logger.info(f"After deduplication: {len(sample_aspects)} aspects")
    
    # Embed
    embedder = AspectEmbedder()
    embeddings = embedder.embed_aspects(sample_aspects)
    
    # Cluster
    clustering_engine = ClusteringEngine()
    labels, metadata = clustering_engine.cluster_aspects(embeddings, sample_aspects, method='kmeans')
    clusters = clustering_engine.organize_clusters(sample_aspects, labels)
    
    # Generate insights
    logger.info(f"\nGenerating insight descriptions for {len(clusters)} clusters...")
    insight_gen = InsightGenerator()
    insights = insight_gen.generate_insights_for_all_clusters(clusters)
    
    # Evaluate
    logger.info("\nEvaluating clustering quality...")
    evaluator = ClusteringEvaluator()
    evaluation = evaluator.generate_evaluation_report(
        clusters, labels, embeddings, insights, metadata
    )
    
    # Display results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"\n📊 Clustering Summary:")
    print(f"   Total aspects: {len(sample_aspects)}")
    print(f"   Number of clusters: {len(clusters)}")
    print(f"   Avg cluster size: {len(sample_aspects) / len(clusters):.1f}")
    
    print(f"\n📈 Clustering Metrics:")
    if 'silhouette_score' in metadata:
        print(f"   Silhouette score: {metadata['silhouette_score']:.4f}")
        print(f"   Davies-Bouldin score: {metadata.get('davies_bouldin_score', 'N/A')}")
    
    print(f"\n✨ Insights Generated:")
    successful = sum(1 for i in insights if i['status'] == 'success')
    print(f"   Success: {successful}/{len(clusters)}")
    
    print(f"\n💰 API Cost (insights generation):")
    cost = insight_gen.get_cost_summary()
    print(f"   Tokens: {cost['total_tokens']:,}")
    print(f"   Estimated cost: ${cost['total_cost_usd']:.4f}")
    
    # Show sample insights
    print(f"\n📋 Sample Insights:")
    for insight_obj in insights[:3]:
        if insight_obj['status'] == 'success':
            cluster_id = insight_obj['cluster_id']
            insight = insight_obj['insight']
            print(f"\n   Cluster {cluster_id}: {insight['cluster_name']}")
            print(f"   {insight['summary']}")
            print(f"   Sentiment: {insight['primary_sentiment']}")
    
    return {
        'aspects': sample_aspects,
        'embeddings': embeddings,
        'clusters': clusters,
        'labels': labels,
        'insights': insights,
        'metadata': metadata,
        'evaluation': evaluation,
        'insight_cost': cost
    }


def run_full_clustering():
    """Run full clustering pipeline on all data."""
    
    print("\n" + "="*70)
    print("STAGE 2: FULL CLUSTERING PIPELINE")
    print("="*70)
    
    start_time = time.time()
    
    # Load and prepare data
    logger.info("\n[1/5] Loading and preparing aspects...")
    aspects, total_comments = AspectProcessor.load_extraction_results()
    
    aspects = AspectProcessor.deduplicate_aspects(aspects)
    aspects = AspectProcessor.normalize_aspect_names(aspects)
    
    logger.info(f"✅ Prepared {len(aspects)} unique aspects from {total_comments} comments")
    
    # Generate embeddings
    logger.info("\n[2/5] Generating embeddings...")
    embedder = AspectEmbedder()
    embeddings = embedder.embed_aspects(aspects)
    
    # Cluster
    logger.info("\n[3/5] Clustering aspects...")
    clustering_engine = ClusteringEngine()
    labels, metadata = clustering_engine.cluster_aspects(embeddings, aspects, method='kmeans')
    clusters = clustering_engine.organize_clusters(aspects, labels)
    
    logger.info(f"✅ Created {len(clusters)} clusters")
    
    # Generate insights
    logger.info("\n[4/5] Generating insights...")
    insight_gen = InsightGenerator()
    insights = insight_gen.generate_insights_for_all_clusters(clusters)
    
    # Evaluate
    logger.info("\n[5/5] Evaluating...")
    evaluator = ClusteringEvaluator()
    evaluation = evaluator.generate_evaluation_report(
        clusters, labels, embeddings, insights, metadata
    )
    
    # Save results
    logger.info("\nSaving results...")
    save_clustering_results(aspects, clusters, labels, insights, metadata, evaluation)
    
    # Final summary
    elapsed = int(time.time() - start_time)
    
    print("\n" + "="*70)
    print("CLUSTERING COMPLETE")
    print("="*70)
    print(f"\n✅ Time: {elapsed//60}m {elapsed%60}s")
    print(f"✅ Clusters: {len(clusters)}")
    print(f"✅ Insights: {sum(1 for i in insights if i['status'] == 'success')}/{len(clusters)}")
    
    cost = insight_gen.get_cost_summary()
    print(f"\n💰 Cost:")
    print(f"   Tokens: {cost['total_tokens']:,}")
    print(f"   Total: ${cost['total_cost_usd']:.4f}")
    
    track_costs(cost, "clustering", len(aspects))
    
    return {
        'aspects': aspects,
        'clusters': clusters,
        'labels': labels,
        'insights': insights,
        'metadata': metadata,
        'evaluation': evaluation
    }


def save_clustering_results(aspects, clusters, labels, insights, metadata, evaluation):
    """Save all clustering results."""
    
    # Prepare clusters data for JSON serialization
    clusters_json = {}
    for cluster_id, cluster_aspects in clusters.items():
        clusters_json[str(cluster_id)] = [
            {
                'aspect': a['aspect'],
                'sentiment': a['sentiment'],
                'confidence': float(a['confidence'])  # Ensure float
            }
            for a in cluster_aspects
        ]
    
    # Main results
    results = {
        'metadata': metadata,
        'clusters': clusters_json,
        'insights': [
            {
                'cluster_id': int(i['cluster_id']),  # Convert to int
                'num_aspects': int(i.get('num_aspects', 0)),  # Convert to int
                'insight': i.get('insight'),
                'status': i.get('status')
            }
            for i in insights
        ],
        'evaluation': evaluation
    }
    
    # Save with custom handling
    with open(CLUSTERING_RESULTS, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Saved clustering results to {CLUSTERING_RESULTS}")
    
    # Insights only (for easy reference)
    insights_only = {}
    for insight_obj in insights:
        if insight_obj['status'] == 'success':
            cluster_id = str(insight_obj['cluster_id'])
            insights_only[cluster_id] = insight_obj['insight']
    
    with open(INSIGHTS_FILE, 'w') as f:
        json.dump(insights_only, f, indent=2, default=str)
    
    logger.info(f"✅ Saved insights to {INSIGHTS_FILE}")
    
    # Evaluation report
    with open(EVALUATION_FILE, 'w') as f:
        json.dump(evaluation, f, indent=2, default=str)
    
    logger.info(f"✅ Saved evaluation to {EVALUATION_FILE}")


def track_costs(cost_summary: Dict, stage: str, num_aspects: int = 0):
    """Track cumulative API costs."""
    
    tracker = {}
    if COST_TRACKER.exists():
        with open(COST_TRACKER, 'r') as f:
            tracker = json.load(f)
    
    tracker['total_spent'] = tracker.get('total_spent', 0) + cost_summary['total_cost_usd']
    tracker['budget_remaining'] = BUDGET_LIMIT - tracker['total_spent']
    tracker['runs'] = tracker.get('runs', [])
    
    tracker['runs'].append({
        'timestamp': datetime.now().isoformat(),
        'stage': stage,
        'aspects_processed': num_aspects,
        'cost': cost_summary['total_cost_usd'],
        'tokens': cost_summary['total_tokens']
    })
    
    with open(COST_TRACKER, 'w') as f:
        json.dump(tracker, f, indent=2)
    
    logger.info(f"\n💰 Cumulative Cost Tracking:")
    logger.info(f"   Total spent: ${tracker['total_spent']:.4f}")
    logger.info(f"   Budget remaining: ${tracker['budget_remaining']:.4f}")
    
    if tracker['budget_remaining'] < 0:
        logger.error(f"⚠️  BUDGET EXCEEDED by ${abs(tracker['budget_remaining']):.2f}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Cluster extracted aspects into insights")
    parser.add_argument('--test', action='store_true', help='Test on 10-20% sample')
    parser.add_argument('--full', action='store_true', help='Run on all aspects')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing results')
    parser.add_argument('--viz', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Verify setup
    if not EXTRACTION_RESULTS.exists():
        logger.error(f"❌ Stage 1 results not found: {EXTRACTION_RESULTS}")
        logger.error("Run Stage 1 first: python src/extract.py --full")
        return False
    
    if args.analyze:
        analyze_existing_results()
    elif args.test:
        test_results = run_test_clustering()
        if args.viz:
            visualize_clustering(test_results)
    elif args.full:
        run_full_clustering()
    else:
        # Default: test first
        logger.info("Running test clustering first...")
        run_test_clustering()
        response = input("\n🤔 Test successful! Run full clustering? (y/n): ").strip().lower()
        if response == 'y':
            run_full_clustering()


def analyze_existing_results():
    """Analyze existing clustering results."""
    logger.info("Analyzing existing results...")
    # Add implementation as needed


if __name__ == "__main__":
    main()