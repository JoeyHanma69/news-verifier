#!/bin/bash

# Training script for ML models using real news data

echo "Starting ML Model Training with Real News Data..."

# Check if Python virtual environment exists
if [ ! -d "python-api/venv" ]; then
    echo "Creating Python virtual environment..."
    cd python-api
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
fi

# Activate virtual environment
source python-api/venv/bin/activate

# Create directories
mkdir -p models
mkdir -p training_data

echo "Training options:"
echo "1. RSS-only training (reliable sources only)"
echo "2. URL-based training (requires URL files)"
echo "3. Mixed training (RSS + unreliable URLs)"

read -p "Choose training method (1-3): " choice

case $choice in
    1)
        echo "Training with RSS feeds from reliable sources..."
        python scripts/ml_training_pipeline.py \
            --method rss \
            --max_articles_per_feed 25 \
            --save_data training_data/rss_training_data.csv
        ;;
    2)
        echo "Training with URL files..."
        
        # Check if URL files exist
        if [ ! -f "reliable_urls.txt" ]; then
            echo "Creating example reliable URLs file..."
            cat > reliable_urls.txt << EOF
# Reliable news URLs for training
https://www.reuters.com/world/
https://apnews.com/
https://www.bbc.com/news
https://www.npr.org/sections/news/
EOF
        fi
        
        if [ ! -f "unreliable_urls.txt" ]; then
            echo "Creating unreliable URLs list..."
            python scripts/create_unreliable_urls.py --output unreliable_urls.txt
        fi
        
        python scripts/ml_training_pipeline.py \
            --method urls \
            --reliable_urls_file reliable_urls.txt \
            --unreliable_urls_file unreliable_urls.txt \
            --save_data training_data/url_training_data.csv
        ;;
    3)
        echo "Mixed training (RSS + unreliable URLs)..."
        
        if [ ! -f "unreliable_urls.txt" ]; then
            echo "Creating unreliable URLs list..."
            python scripts/create_unreliable_urls.py --output unreliable_urls.txt
        fi
        
        python scripts/ml_training_pipeline.py \
            --method mixed \
            --max_articles_per_feed 20 \
            --unreliable_urls_file unreliable_urls.txt \
            --save_data training_data/mixed_training_data.csv \
            --train_transformer
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Training completed!"
echo "Check the following files:"
echo "- models/ directory for trained models"
echo "- training_data/ directory for collected data"
echo "- training_results.json for detailed results"
echo "- model_comparison.png for performance visualization"

# Show results if available
if [ -f "training_results.json" ]; then
    echo ""
    echo "Training Results Summary:"
    python -c "
import json
with open('training_results.json', 'r') as f:
    results = json.load(f)
    print(f'Best Model: {results[\"best_traditional_model\"]}')
    print(f'Accuracy: {results[\"metadata\"][\"accuracy\"]:.4f}')
    print(f'Dataset Size: {results[\"dataset_info\"][\"total_articles\"]} articles')
    print(f'Reliable: {results[\"dataset_info\"][\"reliable_articles\"]}')
    print(f'Unreliable: {results[\"dataset_info\"][\"unreliable_articles\"]}')
"
fi
