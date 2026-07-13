"""
Stage 1: Extract aspects from customer feedback using Claude.

Usage:
    python src/extract.py --test        # Test on 10 comments
    python src/extract.py --full        # Full extraction (5,000 comments)
    python src/extract.py               # Default: test first, then ask to run full
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
from dotenv import load_dotenv
from anthropic import Anthropic
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 500
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024
HAIKU_INPUT_COST = 0.80 / 1_000_000  # \$0.80 per 1M input tokens
HAIKU_OUTPUT_COST = 4.0 / 1_000_000  # \$4.00 per 1M output tokens
BUDGET_LIMIT = 20.0

# Output paths
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

EXTRACTION_RESULTS = OUTPUT_DIR / "01_extraction_results.json"
EXTRACTION_SUMMARY = OUTPUT_DIR / "01_extraction_summary.json"
COST_TRACKER = OUTPUT_DIR / ".cost_tracker.json"


def extract_json_from_response(response_text: str) -> dict:
    """
    Extract JSON from response, handling markdown code block wrapping.
    
    Claude sometimes wraps JSON in ```json ``` blocks.
    This function removes those wrappers and parses the JSON.
    """
    response_text = response_text.strip()
    
    # Remove markdown wrappers
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    response_text = response_text.strip()
    return json.loads(response_text)


class ExtractionEngine:
    """
    LLM-based aspect extraction engine.
    
    Extracts granular aspects (topics, issues, features) from customer feedback
    along with sentiment and confidence scores.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize the Anthropic client."""
        self.client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL")
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.logger = logger
    
    def get_extraction_prompt(self) -> str:
        """Get the extraction prompt template."""
        return """You are an expert at extracting structured insights from customer feedback.

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

Respond ONLY with valid JSON."""
    
    def extract_from_comments(self, comments: List[str]) -> Dict:
        """
        Extract aspects from a list of comments.
        
        Args:
            comments: List of comment strings
            
        Returns:
            Dict with results and cost summary
        """
        results = []
        prompt_template = self.get_extraction_prompt()
        
        for i, comment in enumerate(tqdm(comments, desc="Extracting aspects")):
            try:
                # Format prompt with comment
                prompt = prompt_template.format(comment=comment)
                
                # Call LLM
                message = self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Track tokens
                self.input_tokens += message.usage.input_tokens
                self.output_tokens += message.usage.output_tokens
                
                # Parse response
                response_text = message.content[0].text
                
                try:
                    extracted = extract_json_from_response(response_text)
                    
                    results.append({
                        'idx': i + 1,
                        'comment': comment,
                        'extraction': extracted,
                        'num_aspects': len(extracted.get('aspects', [])),
                        'input_tokens': message.usage.input_tokens,
                        'output_tokens': message.usage.output_tokens,
                        'status': 'success'
                    })
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON parsing failed for comment {i+1}: {e}")
                    results.append({
                        'idx': i + 1,
                        'comment': comment,
                        'status': 'error',
                        'error': f'JSON parsing failed: {str(e)[:100]}'
                    })
                    
            except Exception as e:
                self.logger.error(f"API error for comment {i+1}: {e}")
                results.append({
                    'idx': i + 1,
                    'comment': comment,
                    'status': 'error',
                    'error': str(e)[:100]
                })
        
        return {
            'results': results,
            'cost_summary': self.get_cost_summary()
        }
    
    def get_cost_summary(self) -> dict:
        """Calculate cost based on token usage."""
        input_cost = self.input_tokens * HAIKU_INPUT_COST
        output_cost = self.output_tokens * HAIKU_OUTPUT_COST
        total_cost = input_cost + output_cost
        total_tokens = self.input_tokens + self.output_tokens
        
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': total_tokens,
            'input_cost_usd': round(input_cost, 4),
            'output_cost_usd': round(output_cost, 4),
            'total_cost_usd': round(total_cost, 4),
            'cost_per_token_usd': round(total_cost / total_tokens, 8) if total_tokens > 0 else 0
        }


def load_data() -> pd.DataFrame:
    """Load feedback data."""
    df = pd.read_csv("data/feedback.csv")
    logger.info(f"Loaded {len(df):,} comments from data/feedback.csv")
    return df


def save_extraction_results(all_results: List[Dict], cost_summary: Dict, failed_comments: List[Dict]):
    """Save extraction results to JSON files."""
    
    # Count successful
    successful = len([r for r in all_results if r['status'] == 'success'])
    total_aspects = sum([r.get('num_aspects', 0) for r in all_results if r['status'] == 'success'])
    
    # Save full results
    with open(EXTRACTION_RESULTS, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_comments': len(all_results),
                'successful': successful,
                'failed': len(all_results) - successful,
                'total_aspects': total_aspects,
                'avg_aspects_per_comment': total_aspects / successful if successful > 0 else 0,
                'cost_summary': cost_summary
            },
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"✅ Saved extraction results to {EXTRACTION_RESULTS}")
    
    # Save JSONL for easier processing
    jsonl_file = OUTPUT_DIR / "01_extraction_results.jsonl"
    with open(jsonl_file, 'w') as f:
        for result in all_results:
            if result['status'] == 'success':
                f.write(json.dumps({
                    'comment': result['comment'],
                    'aspects': result['extraction']['aspects']
                }) + '\n')
    
    logger.info(f"✅ Saved JSONL to {jsonl_file}")
    
    # Save summary statistics
    sentiment_dist = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_counts = {}
    
    for result in all_results:
        if result['status'] == 'success':
            for aspect_obj in result['extraction']['aspects']:
                aspect = aspect_obj['aspect']
                sentiment = aspect_obj['sentiment']
                
                sentiment_dist[sentiment] += 1
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
    
    # Get top 20 aspects
    top_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    with open(EXTRACTION_SUMMARY, 'w') as f:
        json.dump({
            'total_aspects_extracted': sum(aspect_counts.values()),
            'sentiment_distribution': sentiment_dist,
            'top_20_aspects': [{'aspect': a, 'count': c} for a, c in top_aspects]
        }, f, indent=2)
    
    logger.info(f"✅ Saved summary to {EXTRACTION_SUMMARY}")
    
    # Save failed comments if any
    if failed_comments:
        failed_file = OUTPUT_DIR / "01_extraction_failed_comments.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_comments, f, indent=2)
        logger.info(f"⚠️ Saved {len(failed_comments)} failed comments to {failed_file}")


def track_costs(cost_summary: Dict, stage: str = "extraction", num_comments: int = 0):
    """Track cumulative costs."""
    
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
        'comments_processed': num_comments,
        'cost': cost_summary['total_cost_usd'],
        'input_tokens': cost_summary['input_tokens'],
        'output_tokens': cost_summary['output_tokens']
    })
    
    with open(COST_TRACKER, 'w') as f:
        json.dump(tracker, f, indent=2)
    
    logger.info(f"\n💰 Cost Tracking:")
    logger.info(f"   Total spent: ${tracker['total_spent']:.4f}")
    logger.info(f"   Budget remaining: ${tracker['budget_remaining']:.4f}")
    
    if tracker['budget_remaining'] < 0:
        logger.error(f"❌ BUDGET EXCEEDED! You've spent ${tracker['total_spent']:.2f} of ${BUDGET_LIMIT:.2f}")
    
    return tracker


def run_test_extraction():
    """Run extraction on a small sample (10 comments)."""
    
    print("\n" + "="*70)
    print("TESTING EXTRACTION ON SMALL SAMPLE (10 comments)")
    print("="*70)
    
    df = load_data()
    test_comments = df['comment'].head(10).tolist()
    
    engine = ExtractionEngine()
    
    logger.info(f"Processing {len(test_comments)} test comments...\n")
    results = engine.extract_from_comments(test_comments)
    
    # Display results
    successful = len([r for r in results['results'] if r['status'] == 'success'])
    total_aspects = sum([r.get('num_aspects', 0) for r in results['results'] if r['status'] == 'success'])
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"\n✅ Success rate: {successful}/{len(test_comments)} ({successful/len(test_comments)*100:.1f}%)")
    print(f"📊 Total aspects extracted: {total_aspects}")
    print(f"📈 Avg aspects per comment: {total_aspects/successful:.2f}" if successful > 0 else "N/A")
    
    print(f"\n💰 Cost Summary:")
    cost = results['cost_summary']
    print(f"   Input tokens: {cost['input_tokens']:,}")
    print(f"   Output tokens: {cost['output_tokens']:,}")
    print(f"   Estimated cost: ${cost['total_cost_usd']:.4f}")
    print(f"   Budget remaining: ${BUDGET_LIMIT - cost['total_cost_usd']:.2f}")
    
    # Show sample extractions
    print(f"\n📝 Sample Extractions:")
    for i, result in enumerate(results['results'][:3], 1):
        if result['status'] == 'success':
            print(f"\n{i}. {result['comment'][:80]}...")
            for aspect in result['extraction']['aspects']:
                print(f"   • {aspect['aspect']}: {aspect['sentiment']}")
    
    return results


def run_full_extraction():
    """Run extraction on all 5,000 comments."""
    
    print("\n" + "="*70)
    print("STAGE 1: FULL EXTRACTION PIPELINE")
    print("="*70)
    
    df = load_data()
    
    print(f"\nProcessing {len(df):,} comments...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated batches: {(len(df) + BATCH_SIZE - 1) // BATCH_SIZE}\n")
    
    engine = ExtractionEngine()
    all_results = []
    failed_comments = []
    start_time = time.time()
    
    # Process in batches
    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(num_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_comments = df.iloc[batch_start:batch_end]['comment'].tolist()
        
        print(f"\n[Batch {batch_num + 1}/{num_batches}] Comments {batch_start + 1:,}-{batch_end:,}")
        
        # Extract
        batch_results = engine.extract_from_comments(batch_comments)
        all_results.extend(batch_results['results'])
        
        # Track failed
        for result in batch_results['results']:
            if result['status'] == 'error':
                failed_comments.append({
                    'idx': result['idx'],
                    'comment': result['comment'][:200],
                    'error': result.get('error', 'unknown')
                })
        
        # Print progress
        successful = len([r for r in batch_results['results'] if r['status'] == 'success'])
        success_rate = successful / len(batch_results['results']) * 100
        
        print(f"  ✅ {successful}/{len(batch_results['results'])} successful ({success_rate:.1f}%)")
        
        cost = batch_results['cost_summary']
        print(f"  💰 Batch cost: ${cost['total_cost_usd']:.4f}")
        print(f"  📊 Cumulative cost: ${engine.get_cost_summary()['total_cost_usd']:.4f}")
        
        elapsed = int(time.time() - start_time)
        print(f"  ⏱️  Elapsed: {elapsed//60}m {elapsed%60}s")
        
        # Budget check
        cumulative = engine.get_cost_summary()['total_cost_usd']
        if cumulative > BUDGET_LIMIT:
            logger.error(f"\n❌ BUDGET EXCEEDED!")
            logger.error(f"   Spent: ${cumulative:.2f}")
            logger.error(f"   Budget: ${BUDGET_LIMIT:.2f}")
            break
    
    # Final summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    
    total_successful = len([r for r in all_results if r['status'] == 'success'])
    total_aspects = sum([r.get('num_aspects', 0) for r in all_results if r['status'] == 'success'])
    
    print(f"\n📊 Results:")
    print(f"   Total processed: {len(all_results):,}")
    print(f"   Successful: {total_successful:,} ({total_successful/len(all_results)*100:.1f}%)")
    print(f"   Failed: {len(all_results) - total_successful}")
    print(f"   Total aspects: {total_aspects:,}")
    print(f"   Avg per comment: {total_aspects/total_successful:.2f}" if total_successful > 0 else "N/A")
    
    final_cost = engine.get_cost_summary()
    print(f"\n💰 Final Cost:")
    print(f"   Input tokens: {final_cost['input_tokens']:,}")
    print(f"   Output tokens: {final_cost['output_tokens']:,}")
    print(f"   Total cost: ${final_cost['total_cost_usd']:.4f}")
    print(f"   Budget remaining: ${BUDGET_LIMIT - final_cost['total_cost_usd']:.2f}")
    
    total_elapsed = int(time.time() - start_time)
    print(f"\n⏱️  Total time: {total_elapsed//3600}h {(total_elapsed%3600)//60}m")
    
    # Save results
    save_extraction_results(all_results, final_cost, failed_comments)
    
    # Track costs
    track_costs(final_cost, "extraction", len(df))
    
    return all_results


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Extract aspects from customer feedback")
    parser.add_argument('--test', action='store_true', help='Run on 10 comments only')
    parser.add_argument('--full', action='store_true', help='Run on all 5,000 comments')
    
    args = parser.parse_args()
    
    # Verify setup
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("❌ ANTHROPIC_API_KEY not set. Run setup.py first.")
        return False
    
    if not Path("data/feedback.csv").exists():
        logger.error("❌ data/feedback.csv not found. Run setup.py first.")
        return False
    
    logger.info("✅ Setup verified")
    
    # Run based on args
    if args.test:
        run_test_extraction()
    elif args.full:
        run_full_extraction()
    else:
        # Default: test first, ask to run full
        test_results = run_test_extraction()
        
        response = input("\n\n🤔 Test successful! Run full extraction on all 5,000 comments? (y/n): ").strip().lower()
        if response == 'y':
            run_full_extraction()
        else:
            logger.info("Aborting. Run 'python src/extract.py --full' when ready.")


if __name__ == "__main__":
    main()