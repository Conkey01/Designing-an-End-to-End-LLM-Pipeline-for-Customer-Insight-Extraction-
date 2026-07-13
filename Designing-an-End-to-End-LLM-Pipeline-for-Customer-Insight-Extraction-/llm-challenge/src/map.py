"""
Stage 3: Map clusters to theme hierarchy.

Usage:
    python src/map.py --semantic    # Semantic-only mapping (no API calls, ~\$0)
    python src/map.py --validate    # Semantic + selective LLM validation (~\$0.30-0.50)
    python src/map.py --full        # Full LLM-based mapping (~\$1.50-2.00, exhaustive)
    python src/map.py --evaluate    # Evaluate existing mapping results
"""

import json
import os
import sys
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic

# Embeddings library
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger_msg = "sentence-transformers not available. Install: pip install sentence-transformers"

# Similarity computation
from sklearn.metrics.pairwise import cosine_similarity

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

CLUSTERING_RESULTS = OUTPUT_DIR / "02_clustering_results.json"
INSIGHTS_FILE = OUTPUT_DIR / "02_insights.json"

MAPPING_RESULTS = OUTPUT_DIR / "03_mapping_results.json"
MAPPING_SUMMARY = OUTPUT_DIR / "03_mapping_summary.json"
THEME_DISTRIBUTION = OUTPUT_DIR / "03_theme_distribution.json"
MAPPING_EVALUATION = OUTPUT_DIR / "03_mapping_evaluation.json"
CONFIDENCE_REPORT = OUTPUT_DIR / "03_confidence_report.json"
COST_TRACKER = OUTPUT_DIR / ".cost_tracker.json"

# Budget
BUDGET_LIMIT = 20.0
HAIKU_INPUT_COST = 0.80 / 1_000_000
HAIKU_OUTPUT_COST = 4.0 / 1_000_000


def load_themes() -> Dict:
    """Load theme hierarchy."""
    with open("themes.json") as f:
        return json.load(f)


def load_insights() -> Dict:
    """Load cluster insights from Stage 2."""
    if not INSIGHTS_FILE.exists():
        raise FileNotFoundError(f"Stage 2 insights not found: {INSIGHTS_FILE}")
    
    with open(INSIGHTS_FILE) as f:
        insights = json.load(f)
    
    logger.info(f"✅ Loaded {len(insights)} cluster insights")
    return insights


def load_clustering_results() -> Dict:
    """Load full clustering results."""
    if not CLUSTERING_RESULTS.exists():
        raise FileNotFoundError(f"Clustering results not found: {CLUSTERING_RESULTS}")
    
    with open(CLUSTERING_RESULTS) as f:
        return json.load(f)


class ThemeMapper:
    """
    Map clusters to theme hierarchy using semantic similarity.
    
    Approach:
    1. Generate embeddings for all themes (one-time, local computation)
    2. For each cluster insight, compute similarity to all themes
    3. Return top-k matches with confidence scores
    4. No API calls needed - entirely local computation
    """
    
    def __init__(self):
        """Initialize embedder and load themes."""
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError("sentence-transformers not available. Install it first.")
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.themes = load_themes()
        self.theme_embeddings = None
        self.theme_list = None
        
        logger.info("✅ ThemeMapper initialized")
    
    def prepare_themes(self) -> None:
        """Generate embeddings for all themes."""
        
        logger.info("\nPreparing theme embeddings...")
        
        # Flatten theme hierarchy
        theme_list = []
        for category in self.themes['themes']:
            for theme in category['themes']:
                theme_list.append({
                    'theme': theme,
                    'category': category['category']
                })
        
        self.theme_list = theme_list
        theme_names = [t['theme'] for t in theme_list]
        
        logger.info(f"Generating embeddings for {len(theme_names)} themes...")
        self.theme_embeddings = self.embedder.encode(theme_names, show_progress_bar=False)
        
        logger.info(f"✅ Theme embeddings ready: {self.theme_embeddings.shape}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text string."""
        return self.embedder.encode([text], show_progress_bar=False)[0]
    
    def find_best_matches(self, insight_text: str, top_k: int = 3) -> List[Dict]:
        """
        Find best matching themes for a cluster insight.
        
        Args:
            insight_text: Cluster insight summary (or name)
            top_k: Number of top matches to return
            
        Returns:
            List of dicts with theme, category, and confidence score
        """
        
        # Embed the insight
        insight_embedding = self.embed_text(insight_text)
        
        # Compute similarity to all themes
        # cosine_similarity returns [[score1, score2, ...]] so index [0]
        similarities = cosine_similarity(
            [insight_embedding],
            self.theme_embeddings
        )[0]
        
        # Get top-k matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        matches = []
        for idx in top_indices:
            theme_info = self.theme_list[idx]
            matches.append({
                'theme': theme_info['theme'],
                'category': theme_info['category'],
                'confidence': float(similarities[idx]),  # 0.0 to 1.0
                'method': 'semantic_similarity'
            })
        
        return matches
    
    def map_all_clusters(self, insights: Dict, top_k: int = 3) -> Dict:
        """
        Map all clusters to themes.
        
        Args:
            insights: Dict of cluster_id -> insight
            top_k: Number of matches per cluster
            
        Returns:
            Mapping results dict
        """
        
        self.prepare_themes()
        
        logger.info(f"\nMapping {len(insights)} clusters to themes (semantic similarity)...")
        
        mapping_results = {}
        
        for cluster_id in tqdm(sorted([int(c) for c in insights.keys()]), 
                              desc="Mapping clusters"):
            cluster_id_str = str(cluster_id)
            insight = insights[cluster_id_str]
            
            # Use cluster name + summary for better matching
            insight_text = f"{insight['cluster_name']}. {insight['summary']}"
            
            # Find matches
            matches = self.find_best_matches(insight_text, top_k=top_k)
            
            # Select primary theme (highest confidence)
            primary_match = matches[0]
            
            mapping_results[cluster_id_str] = {
                'cluster_id': int(cluster_id_str),
                'primary_theme': primary_match['theme'],
                'primary_category': primary_match['category'],
                'primary_confidence': primary_match['confidence'],
                'top_matches': matches,
                'reasoning': f"Semantic similarity between cluster '{insight['cluster_name']}' and theme '{primary_match['theme']}'",
                'status': 'success'
            }
        
        logger.info(f"✅ Mapping complete")
        
        return mapping_results


class MappingValidator:
    """
    Validate and refine mappings using Claude.
    
    Only uses Claude for:
    - Low-confidence cases (semantic similarity < 0.35)
    - Cases that could fit multiple themes
    - Quality assurance on ~10% of mappings
    
    This keeps costs minimal while improving confidence.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize Claude client."""
        # Fix for httpx compatibility: don't pass proxies, use default_headers instead
        self.client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"),
            default_headers={
                "anthropic-version": "2023-06-01"
            }
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.themes = load_themes()
    
    def validate_mapping(self, cluster_id: int, insight: Dict, 
                        semantic_matches: List[Dict]) -> Optional[Dict]:
        """
        Validate mapping using Claude.
        
        Args:
            cluster_id: Cluster ID
            insight: Cluster insight
            semantic_matches: Top matches from semantic similarity
            
        Returns:
            Validated mapping dict or None if no change needed
        """
        
        # Build theme reference
        themes_ref = "AVAILABLE THEMES AND CATEGORIES:\n"
        for category in self.themes['themes']:
            themes_ref += f"\n{category['category']}:\n"
            for theme in category['themes']:
                themes_ref += f"  - {theme}\n"
        
        # Format semantic matches for context
        semantic_context = "\n".join([
            f"  {i+1}. {m['theme']} ({m['category']}): {m['confidence']:.3f}"
            for i, m in enumerate(semantic_matches)
        ])
        
        prompt = f"""You are an expert at mapping customer feedback clusters to business themes.

CLUSTER INSIGHT:
Cluster ID: {cluster_id}
Name: {insight['cluster_name']}
Summary: {insight['summary']}
Primary Sentiment: {insight['primary_sentiment']}
Key Themes: {', '.join(insight['key_themes'])}

SEMANTIC SIMILARITY SUGGESTIONS (from automated matching):
{semantic_context}

{themes_ref}

TASK: Determine the BEST SINGLE THEME mapping for this cluster.
- Choose from the available themes above
- Consider whether the semantic suggestions are appropriate
- Override if a better fit exists based on the insight content
- Provide your reasoning

OUTPUT (JSON only):
{{
  "recommended_theme": "<theme name>",
  "category": "<category name>",
  "confidence": <0.0-1.0>,
  "rationale": "<1-2 sentence explanation>"
}}

Respond ONLY with valid JSON."""
        
        try:
            message = self.client.messages.create(
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.input_tokens += message.usage.input_tokens
            self.output_tokens += message.usage.output_tokens
            
            # Parse response
            response_text = message.content[0].text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            result = json.loads(response_text)
            
            return {
                'recommended_theme': result['recommended_theme'],
                'category': result['category'],
                'confidence': float(result['confidence']),
                'rationale': result['rationale'],
                'validated': True
            }
            
        except Exception as e:
            logger.warning(f"Validation failed for cluster {cluster_id}: {e}")
            return None
    
    def validate_low_confidence_mappings(self, mapping_results: Dict, 
                                        insights: Dict,
                                        confidence_threshold: float = 0.35) -> Dict:
        """
        Validate only low-confidence mappings to improve quality.
        
        Args:
            mapping_results: Results from semantic mapping
            insights: Cluster insights
            confidence_threshold: Only validate below this threshold
            
        Returns:
            Updated mapping results
        """
        
        low_confidence = [
            (cid, result) for cid, result in mapping_results.items()
            if result['primary_confidence'] < confidence_threshold
        ]
        
        if not low_confidence:
            logger.info("✅ No low-confidence mappings to validate")
            return mapping_results
        
        logger.info(f"\nValidating {len(low_confidence)} low-confidence mappings...")
        
        validated_count = 0
        
        for cluster_id, result in tqdm(low_confidence, desc="Validating"):
            insight = insights[cluster_id]
            
            # Validate with Claude
            validation = self.validate_mapping(
                int(cluster_id),
                insight,
                result['top_matches']
            )
            
            if validation:
                # Update mapping with validation
                result['primary_theme'] = validation['recommended_theme']
                result['primary_category'] = validation['category']
                result['primary_confidence'] = validation['confidence']
                result['reasoning'] = validation['rationale']
                result['validated'] = True
                validated_count += 1
        
        logger.info(f"✅ Validated {validated_count}/{len(low_confidence)} mappings")
        
        return mapping_results
    
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


class MappingEvaluator:
    """Evaluate mapping quality and generate reports."""
    
    @staticmethod
    def calculate_coverage(mapping_results: Dict) -> Dict:
        """
        Calculate theme coverage.
        
        Returns:
            Coverage statistics
        """
        
        theme_usage = {}
        category_usage = {}
        
        for result in mapping_results.values():
            theme = result['primary_theme']
            category = result['primary_category']
            
            theme_usage[theme] = theme_usage.get(theme, 0) + 1
            category_usage[category] = category_usage.get(category, 0) + 1
        
        return {
            'themes_used': len(theme_usage),
            'categories_used': len(category_usage),
            'theme_distribution': dict(sorted(
                theme_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'category_distribution': dict(sorted(
                category_usage.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        }
    
    @staticmethod
    def calculate_confidence_stats(mapping_results: Dict) -> Dict:
        """
        Calculate confidence statistics.
        
        Returns:
            Confidence analysis
        """
        
        confidences = [r['primary_confidence'] for r in mapping_results.values()]
        
        return {
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'distribution': {
                'high (>0.5)': int(sum(1 for c in confidences if c > 0.5)),
                'medium (0.35-0.5)': int(sum(1 for c in confidences if 0.35 <= c <= 0.5)),
                'low (<0.35)': int(sum(1 for c in confidences if c < 0.35))
            }
        }
    
    @staticmethod
    def generate_report(mapping_results: Dict, mapping_method: str, 
                       cost_summary: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            mapping_results: Mapping results
            mapping_method: 'semantic', 'validated', or 'full'
            cost_summary: API cost info (if applicable)
            
        Returns:
            Evaluation report
        """
        
        coverage = MappingEvaluator.calculate_coverage(mapping_results)
        confidence = MappingEvaluator.calculate_confidence_stats(mapping_results)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'mapping_method': mapping_method,
            'total_clusters': len(mapping_results),
            'successful_mappings': sum(1 for r in mapping_results.values() if r['status'] == 'success'),
            'coverage': coverage,
            'confidence_statistics': confidence,
            'methodology': {
                'semantic': 'Semantic similarity using sentence-transformers embeddings (no API calls)',
                'validated': 'Semantic similarity + Claude validation for low-confidence cases',
                'full': 'Full Claude-based LLM mapping for all clusters'
            }[mapping_method],
            'recommendations': [
                "Themes with low usage (<3 clusters) may indicate under-coverage or over-specificity",
                "Low confidence mappings (<0.35) should be manually reviewed",
                "Consider broader categories if individual themes are too granular"
            ]
        }
        
        if cost_summary:
            report['api_cost'] = cost_summary
        
        return report


def run_semantic_mapping():
    """Run semantic-only mapping (zero cost)."""
    
    print("\n" + "="*70)
    print("STAGE 3: SEMANTIC MAPPING (NO API CALLS)")
    print("="*70)
    
    # Load data
    insights = load_insights()
    
    # Initialize mapper
    mapper = ThemeMapper()
    
    # Map all clusters
    mapping_results = mapper.map_all_clusters(insights, top_k=3)
    
    # Evaluate
    evaluator = MappingEvaluator()
    evaluation = evaluator.generate_report(mapping_results, 'semantic')
    
    # Save results
    save_mapping_results(mapping_results, evaluation, method='semantic')
    
    # Summary
    print("\n" + "="*70)
    print("SEMANTIC MAPPING COMPLETE")
    print("="*70)
    
    cov = evaluation['coverage']
    conf = evaluation['confidence_statistics']
    
    print(f"\n✅ Mapped {evaluation['successful_mappings']}/{evaluation['total_clusters']} clusters")
    print(f"\n📊 Theme Coverage:")
    print(f"   Unique themes used: {cov['themes_used']}/15")
    print(f"   Unique categories: {cov['categories_used']}/5")
    
    print(f"\n📈 Confidence Distribution:")
    print(f"   High (>0.5): {conf['distribution']['high (>0.5)']}")
    print(f"   Medium (0.35-0.5): {conf['distribution']['medium (0.35-0.5)']}")
    print(f"   Low (<0.35): {conf['distribution']['low (<0.35)']}")
    print(f"   Mean: {conf['mean']:.3f}")
    
    print(f"\n💰 Cost: \$0.00 (no API calls)")
    
    return mapping_results, evaluation


def run_validated_mapping():
    """Run semantic mapping + LLM validation for low-confidence cases."""
    
    print("\n" + "="*70)
    print("STAGE 3: SEMANTIC + VALIDATED MAPPING")
    print("="*70)
    
    # Load data
    insights = load_insights()
    
    # Semantic mapping
    print("\n[1/2] Semantic mapping...")
    mapper = ThemeMapper()
    mapping_results = mapper.map_all_clusters(insights, top_k=3)
    
    # Validation
    print("\n[2/2] Validating low-confidence mappings...")
    validator = MappingValidator()
    mapping_results = validator.validate_low_confidence_mappings(
        mapping_results,
        insights,
        confidence_threshold=0.35
    )
    
    # Evaluate
    cost = validator.get_cost_summary()
    evaluator = MappingEvaluator()
    evaluation = evaluator.generate_report(mapping_results, 'validated', cost)
    
    # Save results
    save_mapping_results(mapping_results, evaluation, method='validated')
    track_costs(cost, 'mapping_validation')
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATED MAPPING COMPLETE")
    print("="*70)
    
    cov = evaluation['coverage']
    conf = evaluation['confidence_statistics']
    
    print(f"\n✅ Mapped {evaluation['successful_mappings']}/{evaluation['total_clusters']} clusters")
    print(f"\n📊 Coverage:")
    print(f"   Themes: {cov['themes_used']}/15 | Categories: {cov['categories_used']}/5")
    
    print(f"\n📈 Confidence:")
    print(f"   Mean: {conf['mean']:.3f} | High (>0.5): {conf['distribution']['high (>0.5)']}")
    
    print(f"\n💰 Cost: ${cost['total_cost_usd']:.4f}")
    print(f"   Tokens: {cost['total_tokens']:,}")
    
    return mapping_results, evaluation


def run_full_mapping():
    """Run full LLM-based mapping (most thorough but higher cost)."""
    
    print("\n" + "="*70)
    print("STAGE 3: FULL LLM-BASED MAPPING")
    print("="*70)
    
    insights = load_insights()
    
    validator = MappingValidator()
    themes = load_themes()
    
    logger.info(f"\nMapping {len(insights)} clusters with Claude...")
    
    mapping_results = {}
    
    for cluster_id_str in tqdm(sorted([c for c in insights.keys() if c.isdigit()], 
                                      key=int),
                               desc="LLM mapping"):
        cluster_id = int(cluster_id_str)
        insight = insights[cluster_id_str]
        
        # Build theme reference
        themes_ref = "AVAILABLE THEMES:\n"
        for category in themes['themes']:
            themes_ref += f"\n{category['category']}:\n"
            for theme in category['themes']:
                themes_ref += f"  - {theme}\n"
        
        prompt = f"""You are an expert at mapping customer feedback clusters to business themes.

CLUSTER INSIGHT:
Cluster ID: {cluster_id}
Name: {insight['cluster_name']}
Summary: {insight['summary']}
Sentiment: {insight['primary_sentiment']}
Key Themes: {', '.join(insight['key_themes'])}

{themes_ref}

TASK: Map this cluster to the SINGLE BEST MATCHING THEME.
- Choose the most appropriate theme from the categories above
- Do NOT invent new themes
- Provide high confidence (0.7-1.0) only if the match is very clear
- Use lower confidence (0.4-0.7) if the match is reasonable but not perfect

OUTPUT (JSON only):
{{
  "theme": "<theme name>",
  "category": "<category name>",
  "confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentence explanation>"
}}

Respond ONLY with valid JSON."""
        
        try:
            message = validator.client.messages.create(
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            
            validator.input_tokens += message.usage.input_tokens
            validator.output_tokens += message.usage.output_tokens
            
            # Parse response
            response_text = message.content[0].text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            result = json.loads(response_text)
            
            mapping_results[cluster_id_str] = {
                'cluster_id': cluster_id,
                'primary_theme': result['theme'],
                'primary_category': result['category'],
                'primary_confidence': float(result['confidence']),
                'reasoning': result['reasoning'],
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Failed to map cluster {cluster_id}: {e}")
            mapping_results[cluster_id_str] = {
                'cluster_id': cluster_id,
                'status': 'error',
                'error': str(e)[:100]
            }
    
    # Evaluate
    cost = validator.get_cost_summary()
    evaluator = MappingEvaluator()
    evaluation = evaluator.generate_report(mapping_results, 'full', cost)
    
    # Save results
    save_mapping_results(mapping_results, evaluation, method='full')
    track_costs(cost, 'mapping_full')
    
    # Summary
    print("\n" + "="*70)
    print("FULL MAPPING COMPLETE")
    print("="*70)
    
    successful = sum(1 for r in mapping_results.values() if r['status'] == 'success')
    print(f"\n✅ Successful: {successful}/{len(mapping_results)}")
    
    if successful > 0:
        cov = evaluation['coverage']
        conf = evaluation['confidence_statistics']
        
        print(f"\n📊 Coverage:")
        print(f"   Themes: {cov['themes_used']}/15 | Categories: {cov['categories_used']}/5")
        
        print(f"\n📈 Confidence:")
        print(f"   Mean: {conf['mean']:.3f}")
        print(f"   High (>0.5): {conf['distribution']['high (>0.5)']}")
    
    print(f"\n💰 Cost: ${cost['total_cost_usd']:.4f}")
    print(f"   Tokens: {cost['total_tokens']:,}")
    
    return mapping_results, evaluation


def save_mapping_results(mapping_results: Dict, evaluation: Dict, method: str = 'semantic'):
    """Save all mapping outputs."""
    
    # Full mapping results
    with open(MAPPING_RESULTS, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'total_clusters': len(mapping_results),
                'successful': sum(1 for r in mapping_results.values() if r['status'] == 'success')
            },
            'mappings': mapping_results
        }, f, indent=2, default=str)
    
    logger.info(f"✅ Saved mapping results to {MAPPING_RESULTS}")
    
    # Theme distribution
    theme_dist = evaluation['coverage']['theme_distribution']
    category_dist = evaluation['coverage']['category_distribution']
    
    with open(THEME_DISTRIBUTION, 'w') as f:
        json.dump({
            'themes': theme_dist,
            'categories': category_dist
        }, f, indent=2)
    
    logger.info(f"✅ Saved theme distribution to {THEME_DISTRIBUTION}")
    
    # Evaluation report
    with open(MAPPING_EVALUATION, 'w') as f:
        json.dump(evaluation, f, indent=2, default=str)
    
    logger.info(f"✅ Saved evaluation to {MAPPING_EVALUATION}")
    
    # Confidence report
    confidence_report = {
        'timestamp': datetime.now().isoformat(),
        'statistics': evaluation['confidence_statistics'],
        'mappings_by_confidence': {
            'high': [
                {
                    'cluster_id': r['cluster_id'],
                    'theme': r['primary_theme'],
                    'confidence': r['primary_confidence']
                }
                for r in mapping_results.values()
                if r.get('status') == 'success' and r['primary_confidence'] > 0.5
            ],
            'medium': [
                {
                    'cluster_id': r['cluster_id'],
                    'theme': r['primary_theme'],
                    'confidence': r['primary_confidence']
                }
                for r in mapping_results.values()
                if r.get('status') == 'success' and 0.35 <= r['primary_confidence'] <= 0.5
            ],
            'low': [
                {
                    'cluster_id': r['cluster_id'],
                    'theme': r['primary_theme'],
                    'confidence': r['primary_confidence']
                }
                for r in mapping_results.values()
                if r.get('status') == 'success' and r['primary_confidence'] < 0.35
            ]
        }
    }
    
    with open(CONFIDENCE_REPORT, 'w') as f:
        json.dump(confidence_report, f, indent=2, default=str)
    
    logger.info(f"✅ Saved confidence report to {CONFIDENCE_REPORT}")


def track_costs(cost_summary: Dict, stage: str = "mapping"):
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
        'cost': cost_summary['total_cost_usd'],
        'tokens': cost_summary['total_tokens']
    })
    
    with open(COST_TRACKER, 'w') as f:
        json.dump(tracker, f, indent=2)
    
    logger.info(f"\n💰 Total Budget Used: ${tracker['total_spent']:.4f} / ${BUDGET_LIMIT:.2f}")
    logger.info(f"   Remaining: ${tracker['budget_remaining']:.4f}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Map clusters to theme hierarchy")
    parser.add_argument('--semantic', action='store_true', 
                       help='Semantic-only (no API calls, \$0)')
    parser.add_argument('--validate', action='store_true',
                       help='Semantic + LLM validation (~\$0.30-0.50)')
    parser.add_argument('--full', action='store_true',
                       help='Full LLM mapping (~\$1.50-2.00)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing mapping')
    
    args = parser.parse_args()
    
    # Verify setup
    if not INSIGHTS_FILE.exists():
        logger.error(f"❌ Stage 2 insights not found: {INSIGHTS_FILE}")
        logger.error("Run Stage 2 first: python src/cluster.py --full")
        return False
    
    if args.semantic:
        run_semantic_mapping()
    elif args.validate:
        run_validated_mapping()
    elif args.full:
        run_full_mapping()
    elif args.evaluate:
        if MAPPING_EVALUATION.exists():
            with open(MAPPING_EVALUATION) as f:
                evaluation = json.load(f)
            print("\n" + "="*70)
            print("MAPPING EVALUATION")
            print("="*70)
            print(json.dumps(evaluation, indent=2))
        else:
            logger.error("No evaluation found. Run mapping first.")
    else:
        # Default: semantic mapping
        logger.info("Running semantic mapping (budget-friendly)...")
        run_semantic_mapping()
        
        response = input("\n🤔 Semantic mapping complete. Validate low-confidence with Claude? (y/n): ")
        if response.strip().lower() == 'y':
            run_validated_mapping()


if __name__ == "__main__":
    main()