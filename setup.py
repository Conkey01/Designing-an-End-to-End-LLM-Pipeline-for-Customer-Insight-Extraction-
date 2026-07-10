"""
Setup script for Chattermill LLM Challenge
Run this once to set everything up: python setup.py
"""

import subprocess
import os
import sys
from pathlib import Path
import shutil

def run_command(cmd, description=""):
    """Run shell command."""
    if description:
        print(description)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        return False
    return True

def main():
    print("""
    Chattermill LLM Challenge - Setup Script \n             
    """)
    
    # Step 1: Install dependencies
    print("\n[1/6] Installing required libraries...")
    !pip install -q anthropic python-dotenv scikit-learn wordcloud pandas numpy matplotlib
    print("Dependencies installed")
    
    # Step 2: Clone the repo
    print("\n[2/6] Cloning repository...")
    if not os.path.exists("llm-challenge"):
        !git clone https://github.com/chattermill/llm-challenge.git
        print("Repository cloned")
    else:
        print("Repository already exists, skipping clone")
    
    # Step 3: Create data directory
    print("\n[3/6] Setting up data directory...")
    Path("data").mkdir(exist_ok=True)
    print("Data directory created")
    
    # Step 4: Copy files from repo to root
    print("\n[4/6] Copying data files...")
    try:
        shutil.copy("llm-challenge/data/feedback.csv", "data/feedback.csv")
        shutil.copy("llm-challenge/themes.json", "themes.json")
        print("Data files copied")
    except FileNotFoundError as e:
        print(f"Error copying files: {e}")
        print("   Make sure llm-challenge/ directory exists")
        return False
    
    # Step 5: Create output/src directories
    print("\n[5/6] Creating project directories...")
    Path("outputs").mkdir(exist_ok=True)
    Path("src").mkdir(exist_ok=True)
    print("Directories created")
    
    # Step 6: Set up API credentials
    print("\n[6/6] Setting up API credentials...")
    api_key = input("\n Enter your ANTHROPIC_API_KEY (sk-...): ").strip()
    base_url = "https://llm-api.datascience.chattermill.xyz/anthropic"
    
    # Save to .env file
    env_content = f"""ANTHROPIC_API_KEY={api_key}
ANTHROPIC_BASE_URL={base_url}
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("API credentials saved to .env")
    
    # Verify setup
    print("VERIFYING SETUP\n")
    
    # Check files
    checks = {
        "data/feedback.csv": os.path.exists("data/feedback.csv"),
        "themes.json": os.path.exists("themes.json"),
        ".env file": os.path.exists(".env"),
        "outputs/ directory": os.path.isdir("outputs"),
        "src/ directory": os.path.isdir("src"),
    }
    
    for check, status in checks.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check}")
    
    # Test API connection
    print("TESTING API CONNECTION\n")
    
    try:
        import os
        os.environ['ANTHROPIC_API_KEY'] = api_key
        os.environ['ANTHROPIC_BASE_URL'] = base_url
        
        from anthropic import Anthropic
        client = Anthropic(
            api_key=api_key,
            base_url=base_url
        )
        
        models = client.models.list()
        print(f"API connection successful!")
        print(f"   Available models: {len(models.data)}")
        
    except Exception as e:
        print(f" API connection failed: {e}")
        print("   Check your API key is correct")
        return False
    
    print("SETUP COMPLETE!\n")
    print("\nNext steps:")
    print("1. Load and explore data:")
    print("   jupyter notebook notebooks/01_data_exploration.ipynb")
    print("\n2. Run extraction:")
    print("   python src/extract.py")
    print("   or")
    print("   python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
