"""
Setup script for Chattermill LLM Challenge
Run this once: python setup.py
(Make sure you're already in the llm-challenge directory)
"""

import subprocess
import os
import sys
from pathlib import Path
import shutil

def main():
    print("""
╔════════════════════════════════════════════════════════╗
║  Chattermill LLM Challenge - Setup Script              ║
╚════════════════════════════════════════════════════════╝
    """)
    
    # Check if we're in the right directory
    if not os.path.exists("data/feedback.csv") and os.path.exists("../llm-challenge/data/feedback.csv"):
        print("❌ Error: You need to cd into llm-challenge first!")
        print("\nRun these commands:")
        print("  git clone https://github.com/chattermill/llm-challenge.git")
        print("  cd llm-challenge")
        print("  python setup.py")
        return False
    
    # Step 1: Install dependencies
    print("\n[1/5] Installing required libraries...")
    result = subprocess.run(
        "pip install -q anthropic python-dotenv scikit-learn wordcloud pandas numpy matplotlib",
        shell=True
    )
    if result.returncode == 0:
        print("✅ Dependencies installed")
    else:
        print("❌ Failed to install dependencies")
        return False
    
    # Step 2: Create output/src directories
    print("\n[2/5] Creating project directories...")
    Path("outputs").mkdir(exist_ok=True)
    Path("src").mkdir(exist_ok=True)
    Path("notebooks").mkdir(exist_ok=True)
    print("✅ Directories created")
    
    # Step 3: Verify data files exist
    print("\n[3/5] Verifying data files...")
    if not os.path.exists("data/feedback.csv"):
        print("❌ data/feedback.csv not found")
        return False
    if not os.path.exists("themes.json"):
        print("❌ themes.json not found")
        return False
    print("✅ Data files verified")
    
    # Step 4: Set up API credentials
    print("\n[4/5] Setting up API credentials...")
    api_key = input("\n🔑 Enter your ANTHROPIC_API_KEY (sk-...): ").strip()
    base_url = "https://llm-api.datascience.chattermill.xyz/anthropic"
    
    # Save to .env file
    env_content = f"""ANTHROPIC_API_KEY={api_key}
ANTHROPIC_BASE_URL={base_url}
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ API credentials saved to .env")
    
    # Step 5: Test API connection
    print("\n[5/5] Testing API connection...")
    
    try:
        os.environ['ANTHROPIC_API_KEY'] = api_key
        os.environ['ANTHROPIC_BASE_URL'] = base_url
        
        from anthropic import Anthropic
        client = Anthropic(
            api_key=api_key,
            base_url=base_url
        )
        
        models = client.models.list()
        print(f"✅ API connection successful!")
        print(f"   {len(models.data)} models available")
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("   Check your API key is correct")
        return False
    
    # Final verification
    print("\n" + "="*60)
    print("SETUP VERIFICATION")
    print("="*60)
    
    checks = {
        "data/feedback.csv": os.path.exists("data/feedback.csv"),
        "themes.json": os.path.exists("themes.json"),
        ".env file": os.path.exists(".env"),
        "outputs/ directory": os.path.isdir("outputs"),
        "src/ directory": os.path.isdir("src"),
        "notebooks/ directory": os.path.isdir("notebooks"),
    }
    
    for check, status in checks.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check}")
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nYou can now:")
    print("1. Load data:")
    print("   python load_data.py")
    print("\n2. Explore data (optional):")
    print("   jupyter notebook notebooks/01_data_exploration.ipynb")
    print("\n3. Run extraction:")
    print("   python src/extract.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)