#!/usr/bin/env python3
"""
Machine Learning Training Pipeline for News Verification - URL-Based Training
Run with: python scripts/ml_training_pipeline.py --source rss --reliable_feeds feeds.txt
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import os
import json
import argparse
from datetime import datetime, timedelta
import requests
import feedparser
from newspaper import Article
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from urllib.parse import urlparse
import time
import logging
from typing import List, Dict, Tuple
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class URLBasedDataCollector:
    """Collect training data from real news sources via URLs and RSS feeds"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Known reliable sources with RSS feeds
        self.reliable_sources = {
            'reuters': {
                'rss_feeds': [
                    'http://feeds.reuters.com/reuters/topNews',
                    'http://feeds.reuters.com/reuters/worldNews',
                    'http://feeds.reuters.com/reuters/businessNews'
                ],
                'label': 0  # 0 = reliable
            },
            'ap_news': {
                'rss_feeds': [
                    'https://feeds.apnews.com/rss/apf-topnews',
                    'https://feeds.apnews.com/rss/apf-usnews',
                    'https://feeds.apnews.com/rss/apf-worldnews'
                ],
                'label': 0
            },
            'bbc': {
                'rss_feeds': [
                    'http://feeds.bbci.co.uk/news/rss.xml',
                    'http://feeds.bbci.co.uk/news/world/rss.xml',
                    'http://feeds.bbci.co.uk/news/business/rss.xml'
                ],
                'label': 0
            },
            'npr': {
                'rss_feeds': [
                    'https://feeds.npr.org/1001/rss.xml',
                    'https://feeds.npr.org/1004/rss.xml',
                    'https://feeds.npr.org/1003/rss.xml'
                ],
                'label': 0
            }
        }
        
        # Known unreliable sources (you would add actual unreliable RSS feeds here)
        self.unreliable_sources = {
            'example_unreliable': {
                'rss_feeds': [
                    # Add actual unreliable source RSS feeds here
                    # For demonstration, we'll use a different approach
                ],
                'label': 1  # 1 = unreliable
            }
        }
    
    def extract_article_content(self, url: str) -> Dict:
        """Extract article content from URL"""
        try:
            # Method 1: Use newspaper3k
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 100:
                return {
                    'url': url,
                    'title': article.title,
                    'text': article.text,
                    'authors': article.authors,
                    'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                    'extraction_method': 'newspaper3k',
                    'success': True
                }
        except Exception as e:
            logger.warning(f"Newspaper3k failed for {url}: {e}")
        
        # Method 2: BeautifulSoup fallback
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract main content
            content_selectors = [
                'article', '.article-content', '.post-content', 
                '.entry-content', '.content', 'main', '.story-body'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            if content and len(content) > 100:
                return {
                    'url': url,
                    'title': title_text,
                    'text': content,
                    'authors': [],
                    'publish_date': None,
                    'extraction_method': 'beautifulsoup',
                    'success': True
                }
                
        except Exception as e:
            logger.warning(f"BeautifulSoup failed for {url}: {e}")
        
        return {'url': url, 'success': False, 'error': 'Content extraction failed'}
    
    def collect_from_rss_feeds(self, max_articles_per_feed: int = 20) -> List[Dict]:
        """Collect articles from RSS feeds of reliable sources"""
        collected_articles = []
        
        logger.info("Collecting articles from reliable RSS feeds...")
        
        for source_name, source_info in self.reliable_sources.items():
            logger.info(f"Processing {source_name}...")
            
            for feed_url in source_info['rss_feeds']:
                try:
                    logger.info(f"Fetching RSS feed: {feed_url}")
                    feed = feedparser.parse(feed_url)
                    
                    articles_from_feed = 0
                    for entry in feed.entries:
                        if articles_from_feed >= max_articles_per_feed:
                            break
                        
                        # Extract article content
                        article_data = self.extract_article_content(entry.link)
                        
                        if article_data.get('success', False):
                            article_data.update({
                                'source': source_name,
                                'label': source_info['label'],
                                'feed_url': feed_url,
                                'rss_title': entry.title,
                                'rss_published': entry.get('published', ''),
                                'collection_timestamp': datetime.now().isoformat()
                            })
                            
                            collected_articles.append(article_data)
                            articles_from_feed += 1
                            
                            # Add delay to be respectful to servers
                            time.sleep(1)
                        
                        else:
                            logger.warning(f"Failed to extract content from {entry.link}")
                
                except Exception as e:
                    logger.error(f"Error processing RSS feed {feed_url}: {e}")
        
        logger.info(f"Collected {len(collected_articles)} articles from RSS feeds")
        return collected_articles
    
    def collect_from_url_list(self, url_file: str, label: int) -> List[Dict]:
        """Collect articles from a list of URLs with specified label"""
        collected_articles = []
        
        try:
            with open(url_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            logger.info(f"Processing {len(urls)} URLs from {url_file}")
            
            for url in urls:
                article_data = self.extract_article_content(url)
                
                if article_data.get('success', False):
                    article_data.update({
                        'source': 'url_list',
                        'label': label,
                        'collection_timestamp': datetime.now().isoformat()
                    })
                    
                    collected_articles.append(article_data)
                    
                    # Add delay
                    time.sleep(1)
                else:
                    logger.warning(f"Failed to extract content from {url}")
        
        except FileNotFoundError:
            logger.error(f"URL file {url_file} not found")
        except Exception as e:
            logger.error(f"Error processing URL file {url_file}: {e}")
        
        logger.info(f"Collected {len(collected_articles)} articles from URL list")
        return collected_articles
    
    def collect_from_search_results(self, search_terms: List[str], reliable: bool = True) -> List[Dict]:
        """Collect articles by searching reliable news sources"""
        collected_articles = []
        
        # This would integrate with news APIs like NewsAPI, but for now we'll use RSS
        # In production, you would use:
        # - NewsAPI (newsapi.org)
        # - Google News API
        # - Bing News API
        # - Custom web scraping of search results
        
        logger.info(f"Search-based collection not implemented yet. Use RSS feeds or URL lists instead.")
        return collected_articles
    
    def save_collected_data(self, articles: List[Dict], output_file: str):
        """Save collected articles to file"""
        try:
            # Create DataFrame
            df_data = []
            for article in articles:
                if article.get('success', False):
                    df_data.append({
                        'url': article['url'],
                        'title': article['title'],
                        'text': article['text'],
                        'label': article['label'],
                        'source': article['source'],
                        'authors': str(article.get('authors', [])),
                        'publish_date': article.get('publish_date', ''),
                        'collection_timestamp': article['collection_timestamp']
                    })
            
            df = pd.DataFrame(df_data)
            
            # Save to different formats
            if output_file.endswith('.csv'):
                df.to_csv(output_file, index=False)
            elif output_file.endswith('.json'):
                df.to_json(output_file, orient='records', indent=2)
            else:
                # Default to CSV
                df.to_csv(output_file + '.csv', index=False)
            
            logger.info(f"Saved {len(df)} articles to {output_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error saving collected data: {e}")
            return None

class MLTrainingPipeline:
    def __init__(self):
        self.data_collector = URLBasedDataCollector()
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        
    def collect_training_data(self, collection_method: str, **kwargs) -> pd.DataFrame:
        """Collect training data using specified method"""
        
        if collection_method == 'rss':
            # Collect from RSS feeds
            max_articles = kwargs.get('max_articles_per_feed', 20)
            articles = self.data_collector.collect_from_rss_feeds(max_articles)
            
        elif collection_method == 'urls':
            # Collect from URL lists
            reliable_urls = kwargs.get('reliable_urls_file')
            unreliable_urls = kwargs.get('unreliable_urls_file')
            
            articles = []
            
            if reliable_urls:
                reliable_articles = self.data_collector.collect_from_url_list(reliable_urls, 0)
                articles.extend(reliable_articles)
            
            if unreliable_urls:
                unreliable_articles = self.data_collector.collect_from_url_list(unreliable_urls, 1)
                articles.extend(unreliable_articles)
                
        elif collection_method == 'mixed':
            # Combine RSS and URL methods
            articles = []
            
            # RSS feeds
            rss_articles = self.data_collector.collect_from_rss_feeds(
                kwargs.get('max_articles_per_feed', 15)
            )
            articles.extend(rss_articles)
            
            # URL lists if provided
            if kwargs.get('unreliable_urls_file'):
                unreliable_articles = self.data_collector.collect_from_url_list(
                    kwargs['unreliable_urls_file'], 1
                )
                articles.extend(unreliable_articles)
        
        else:
            raise ValueError(f"Unknown collection method: {collection_method}")
        
        # Convert to DataFrame
        if articles:
            df_data = []
            for article in articles:
                if article.get('success', False):
                    df_data.append({
                        'url': article['url'],
                        'title': article['title'],
                        'text': article['text'],
                        'label': article['label'],
                        'source': article['source'],
                        'word_count': len(article['text'].split()),
                        'collection_timestamp': article['collection_timestamp']
                    })
            
            df = pd.DataFrame(df_data)
            logger.info(f"Created dataset with {len(df)} articles")
            return df
        else:
            logger.error("No articles collected")
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the collected data"""
        logger.info("Preprocessing data...")
        
        # Remove duplicates based on text content
        df['text_hash'] = df['text'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        df = df.drop_duplicates(subset=['text_hash'])
        df = df.drop('text_hash', axis=1)
        
        # Filter by minimum word count
        min_words = 50
        df = df[df['word_count'] >= min_words]
        
        # Clean text
        df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
        df['text'] = df['text'].str.lower()
        
        # Balance dataset if needed
        label_counts = df['label'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        # If dataset is heavily imbalanced, balance it
        if len(label_counts) > 1:
            min_count = min(label_counts.values())
            if max(label_counts.values()) > min_count * 3:  # If 3:1 ratio or worse
                logger.info("Balancing dataset...")
                balanced_dfs = []
                for label in label_counts.index:
                    label_df = df[df['label'] == label].sample(n=min(min_count * 2, len(df[df['label'] == label])))
                    balanced_dfs.append(label_df)
                df = pd.concat(balanced_dfs, ignore_index=True)
        
        logger.info(f"Preprocessed dataset: {len(df)} articles")
        return df
    
    def train_traditional_models(self, X_train, X_test, y_train, y_test):
        """Train traditional ML models"""
        logger.info("Training traditional ML models...")
        
        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(
            max_features=10000, 
            stop_words='english', 
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        self.vectorizers['tfidf'] = tfidf
        
        # Models to train
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42, kernel='rbf')
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_tfidf, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_tfidf)
            y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(classification_report(y_test, y_pred))
            print("-" * 50)
        
        self.models.update(results)
        return results
    
    def train_transformer_model(self, df: pd.DataFrame):
        """Train a transformer-based model"""
        logger.info("Training transformer model...")
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
        )
        
        # Initialize tokenizer and model
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        
        # Create datasets
        train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
        test_dataset = NewsDataset(test_texts, test_labels, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=100,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        os.makedirs('./models', exist_ok=True)
        model.save_pretrained('./models/transformer_fake_news')
        tokenizer.save_pretrained('./models/transformer_fake_news')
        
        logger.info("Transformer model training completed")
        return model, tokenizer
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate and compare all models"""
        logger.info("Evaluating models...")
        
        comparison_data = []
        
        for name, result in self.models.items():
            if 'accuracy' in result:
                comparison_data.append({
                    'Model': name,
                    'Accuracy': result['accuracy'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Comparison:")
        print("-" * 60)
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        if len(comparison_data) > 0:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.bar(comparison_df['Model'], comparison_df['Accuracy'])
            plt.title('Model Accuracy Comparison')
            plt.xticks(rotation=45)
            plt.ylabel('Accuracy')
            
            plt.subplot(1, 2, 2)
            plt.bar(comparison_df['Model'], comparison_df['CV Mean'])
            plt.title('Cross-Validation Score Comparison')
            plt.xticks(rotation=45)
            plt.ylabel('CV Score')
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Model comparison plot saved as 'model_comparison.png'")
        
        return comparison_df
    
    def save_best_model(self):
        """Save the best performing model"""
        if not self.models:
            logger.error("No models trained yet")
            return None, None
        
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x].get('accuracy', 0))
        best_model = self.models[best_model_name]['model']
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(best_model, f'models/best_model_{best_model_name}.joblib')
        joblib.dump(self.vectorizers['tfidf'], 'models/tfidf_vectorizer.joblib')
        
        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'accuracy': self.models[best_model_name]['accuracy'],
            'cv_mean': self.models[best_model_name]['cv_mean'],
            'cv_std': self.models[best_model_name]['cv_std'],
            'training_date': datetime.now().isoformat(),
            'model_type': 'traditional_ml',
            'data_collection_method': 'url_based'
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Best model ({best_model_name}) saved with accuracy: {metadata['accuracy']:.4f}")
        
        return best_model_name, metadata
    
    def run_full_pipeline(self, collection_method: str, **kwargs):
        """Run the complete training pipeline"""
        logger.info("Starting URL-based ML Training Pipeline...")
        logger.info("=" * 60)
        
        # Collect data
        logger.info(f"Collecting data using method: {collection_method}")
        df = self.collect_training_data(collection_method, **kwargs)
        
        if df.empty:
            logger.error("No data collected. Cannot proceed with training.")
            return None
        
        # Save collected data
        output_file = kwargs.get('save_data', 'collected_training_data.csv')
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Raw collected data saved to {output_file}")
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        if len(df) < 10:
            logger.error("Insufficient data after preprocessing. Need at least 10 articles.")
            return None
        
        # Check if we have both labels
        unique_labels = df['label'].unique()
        if len(unique_labels) < 2:
            logger.error(f"Need both reliable (0) and unreliable (1) articles. Found labels: {unique_labels}")
            return None
        
        # Split data
        X = df['text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} articles")
        logger.info(f"Test set: {len(X_test)} articles")
        
        # Train traditional models
        self.train_traditional_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Save best model
        best_model, metadata = self.save_best_model()
        
        # Train transformer model if requested and sufficient data
        if kwargs.get('train_transformer', False) and len(df) >= 100:
            try:
                transformer_model, tokenizer = self.train_transformer_model(df)
                logger.info("Transformer model training completed!")
            except Exception as e:
                logger.error(f"Transformer training failed: {e}")
        
        logger.info("=" * 60)
        logger.info("Training pipeline completed!")
        
        return {
            'best_traditional_model': best_model,
            'metadata': metadata,
            'results': self.models,
            'dataset_info': {
                'total_articles': len(df),
                'reliable_articles': len(df[df['label'] == 0]),
                'unreliable_articles': len(df[df['label'] == 1]),
                'sources': df['source'].unique().tolist()
            }
        }

def main():
    parser = argparse.ArgumentParser(description='Train ML models using real news data from URLs')
    parser.add_argument('--method', choices=['rss', 'urls', 'mixed'], default='rss',
                       help='Data collection method')
    parser.add_argument('--max_articles_per_feed', type=int, default=20,
                       help='Maximum articles to collect per RSS feed')
    parser.add_argument('--reliable_urls_file', 
                       help='File containing URLs of reliable news articles')
    parser.add_argument('--unreliable_urls_file',
                       help='File containing URLs of unreliable news articles')
    parser.add_argument('--save_data', default='collected_training_data.csv',
                       help='File to save collected data')
    parser.add_argument('--train_transformer', action='store_true',
                       help='Also train transformer model (requires more data and resources)')
    parser.add_argument('--output_dir', default='models',
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.method == 'urls' and not (args.reliable_urls_file or args.unreliable_urls_file):
        parser.error("URLs method requires --reliable_urls_file and/or --unreliable_urls_file")
    
    if args.method == 'mixed' and not args.unreliable_urls_file:
        logger.warning("Mixed method without unreliable URLs will only use RSS feeds (reliable sources)")
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    
    # Prepare kwargs
    kwargs = {
        'max_articles_per_feed': args.max_articles_per_feed,
        'save_data': args.save_data,
        'train_transformer': args.train_transformer
    }
    
    if args.reliable_urls_file:
        kwargs['reliable_urls_file'] = args.reliable_urls_file
    
    if args.unreliable_urls_file:
        kwargs['unreliable_urls_file'] = args.unreliable_urls_file
    
    # Run training
    try:
        results = pipeline.run_full_pipeline(args.method, **kwargs)
        
        if results:
            print("\n" + "="*60)
            print("TRAINING RESULTS SUMMARY")
            print("="*60)
            print(f"Best model: {results['best_traditional_model']}")
            print(f"Accuracy: {results['metadata']['accuracy']:.4f}")
            print(f"Dataset size: {results['dataset_info']['total_articles']} articles")
            print(f"Reliable articles: {results['dataset_info']['reliable_articles']}")
            print(f"Unreliable articles: {results['dataset_info']['unreliable_articles']}")
            print(f"Sources: {', '.join(results['dataset_info']['sources'])}")
            
            # Save results summary
            with open('training_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("Training results saved to training_results.json")
        else:
            logger.error("Training failed")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
