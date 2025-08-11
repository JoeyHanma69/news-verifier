from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import aiohttp
import numpy as np
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
import re
from datetime import datetime, timedelta
import hashlib
import json
from transformers import pipeline
import torch
import cv2
from PIL import Image
import io
import whois
import dns.resolver
import ssl
import socket
from newspaper import Article
import feedparser
import time

app = FastAPI(title="AI News Verification Python API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained models for real-time analysis
class MLModels:
    def __init__(self):
        self.sentiment_analyzer = None
        self.bias_detector = None
        self.load_models()
    
    def load_models(self):
        try:
            # Load sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Load bias detection model
            self.bias_detector = pipeline(
                "text-classification",
                model="unitary/toxic-bert"
            )
            
        except Exception as e:
            print(f"Error loading models: {e}")

ml_models = MLModels()

# Known reliable and unreliable sources
RELIABLE_SOURCES = {
    'reuters.com': {'score': 95, 'type': 'news_agency'},
    'ap.org': {'score': 95, 'type': 'news_agency'},
    'bbc.com': {'score': 90, 'type': 'public_broadcaster'},
    'npr.org': {'score': 88, 'type': 'public_radio'},
    'pbs.org': {'score': 87, 'type': 'public_broadcaster'},
    'nytimes.com': {'score': 85, 'type': 'newspaper'},
    'washingtonpost.com': {'score': 85, 'type': 'newspaper'},
    'wsj.com': {'score': 84, 'type': 'newspaper'},
    'theguardian.com': {'score': 82, 'type': 'newspaper'},
    'cnn.com': {'score': 78, 'type': 'tv_news'},
    'politico.com': {'score': 82, 'type': 'political_news'},
    'axios.com': {'score': 80, 'type': 'digital_news'}
}

UNRELIABLE_SOURCES = {
    'infowars.com': {'score': 10, 'reason': 'conspiracy_theories'},
    'naturalnews.com': {'score': 15, 'reason': 'pseudoscience'},
    'breitbart.com': {'score': 25, 'reason': 'extreme_bias'},
    'rt.com': {'score': 30, 'reason': 'state_propaganda'},
    'sputniknews.com': {'score': 30, 'reason': 'state_propaganda'}
}

# Pydantic models
class ArticleAnalysisRequest(BaseModel):
    text: str
    url: Optional[str] = None
    include_ml_analysis: bool = True

class ImageVerificationRequest(BaseModel):
    image_url: str
    article_text: Optional[str] = None

class MLTrainingRequest(BaseModel):
    articles: List[dict]
    labels: List[str]

class BiasAnalysisResponse(BaseModel):
    bias_score: float
    bias_type: str
    confidence: float
    explanation: str

class SentimentAnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    emotional_indicators: List[str]

class MLAnalysisResponse(BaseModel):
    bias_analysis: BiasAnalysisResponse
    sentiment_analysis: SentimentAnalysisResponse
    fake_probability: float
    linguistic_features: dict
    readability_score: float

class URLAnalysisRequest(BaseModel):
    url: str
    deep_analysis: bool = True

class BatchURLRequest(BaseModel):
    urls: List[str]
    max_concurrent: int = 5

class SourceAnalysisRequest(BaseModel):
    domain: str

class CrossReferenceRequest(BaseModel):
    article_url: str
    claim: str

# Web scraping service
class WebScrapingService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def extract_article_content(self, url: str) -> Dict:
        """Extract article content from URL using multiple methods"""
        try:
            # Method 1: Use newspaper3k library
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 100:
                return {
                    'title': article.title,
                    'text': article.text,
                    'authors': article.authors,
                    'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                    'top_image': article.top_image,
                    'meta_keywords': article.meta_keywords,
                    'meta_description': article.meta_description,
                    'extraction_method': 'newspaper3k'
                }
        except Exception as e:
            print(f"Newspaper3k extraction failed: {e}")
        
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
            
            # Extract main content using common selectors
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
            
            # If no specific content found, get all paragraphs
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Extract metadata
            meta_description = soup.find('meta', attrs={'name': 'description'})
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            
            return {
                'title': title_text,
                'text': content,
                'authors': [],
                'publish_date': None,
                'top_image': None,
                'meta_keywords': meta_keywords.get('content', '') if meta_keywords else '',
                'meta_description': meta_description.get('content', '') if meta_description else '',
                'extraction_method': 'beautifulsoup'
            }
            
        except Exception as e:
            raise Exception(f"Failed to extract content: {str(e)}")

# Source credibility analyzer
class SourceCredibilityAnalyzer:
    def __init__(self):
        self.web_scraper = WebScrapingService()
    
    def analyze_domain_credibility(self, url: str) -> Dict:
        """Analyze domain credibility using multiple factors"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('www.', '').lower()
            
            # Check against known sources
            if domain in RELIABLE_SOURCES:
                return {
                    'domain': domain,
                    'credibility_score': RELIABLE_SOURCES[domain]['score'],
                    'source_type': RELIABLE_SOURCES[domain]['type'],
                    'status': 'known_reliable'
                }
            
            if domain in UNRELIABLE_SOURCES:
                return {
                    'domain': domain,
                    'credibility_score': UNRELIABLE_SOURCES[domain]['score'],
                    'reason': UNRELIABLE_SOURCES[domain]['reason'],
                    'status': 'known_unreliable'
                }
            
            # Analyze unknown domain
            domain_analysis = self.analyze_unknown_domain(domain)
            
            return {
                'domain': domain,
                'credibility_score': domain_analysis['score'],
                'analysis': domain_analysis,
                'status': 'analyzed'
            }
            
        except Exception as e:
            return {
                'domain': 'unknown',
                'credibility_score': 50,
                'error': str(e),
                'status': 'error'
            }
    
    def analyze_unknown_domain(self, domain: str) -> Dict:
        """Analyze unknown domain for credibility indicators"""
        score = 50  # Start with neutral score
        analysis = {}
        
        try:
            # WHOIS analysis
            whois_info = whois.whois(domain)
            if whois_info.creation_date:
                if isinstance(whois_info.creation_date, list):
                    creation_date = whois_info.creation_date[0]
                else:
                    creation_date = whois_info.creation_date
                
                domain_age = (datetime.now() - creation_date).days
                analysis['domain_age_days'] = domain_age
                
                # Older domains are generally more credible
                if domain_age > 365 * 5:  # 5+ years
                    score += 15
                elif domain_age > 365 * 2:  # 2+ years
                    score += 10
                elif domain_age < 90:  # Less than 3 months
                    score -= 20
            
        except Exception as e:
            analysis['whois_error'] = str(e)
        
        try:
            # SSL certificate check
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    analysis['ssl_valid'] = True
                    score += 5
        except Exception:
            analysis['ssl_valid'] = False
            score -= 10
        
        # Domain name analysis
        suspicious_patterns = [
            r'\d{4,}',  # Many numbers
            r'[a-z]{20,}',  # Very long words
            r'(fake|hoax|conspiracy|truth|exposed)',  # Suspicious keywords
            r'(news|media|press|times|post|herald)(\d+|[a-z]{1,3})$'  # Fake news patterns
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain, re.IGNORECASE):
                score -= 10
                analysis['suspicious_domain_pattern'] = True
                break
        
        # TLD analysis
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            score -= 15
            analysis['suspicious_tld'] = True
        
        analysis['final_score'] = max(0, min(100, score))
        return analysis

# Content analyzer
class ContentAnalyzer:
    def __init__(self):
        self.ml_models = ml_models
    
    def analyze_article_content(self, content: Dict) -> Dict:
        """Comprehensive content analysis"""
        text = content.get('text', '')
        title = content.get('title', '')
        
        if not text or len(text) < 50:
            return {'error': 'Insufficient content for analysis'}
        
        analysis = {}
        
        # Basic text statistics
        analysis['text_stats'] = self.calculate_text_statistics(text)
        
        # Sentiment analysis
        analysis['sentiment'] = self.analyze_sentiment(text)
        
        # Bias detection
        analysis['bias'] = self.detect_bias(text)
        
        # Clickbait detection
        analysis['clickbait'] = self.detect_clickbait(title, text)
        
        # Emotional manipulation detection
        analysis['emotional_manipulation'] = self.detect_emotional_manipulation(text)
        
        # Citation analysis
        analysis['citations'] = self.analyze_citations(text)
        
        # Language quality assessment
        analysis['language_quality'] = self.assess_language_quality(text)
        
        return analysis
    
    def calculate_text_statistics(self, text: str) -> Dict:
        """Calculate basic text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'exclamation_ratio': text.count('!') / len(text) if text else 0,
            'question_ratio': text.count('?') / len(text) if text else 0,
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using transformer model"""
        try:
            # Limit text length for model
            text_sample = text[:512]
            result = self.ml_models.sentiment_analyzer(text_sample)[0]
            
            return {
                'label': result['label'],
                'confidence': result['score'],
                'analysis': f"Text shows {result['label']} sentiment with {result['score']:.2%} confidence"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_bias(self, text: str) -> Dict:
        """Detect bias using transformer model"""
        try:
            text_sample = text[:512]
            result = self.ml_models.bias_detector(text_sample)[0]
            
            bias_score = result['score'] if result['label'] == 'TOXIC' else 1 - result['score']
            
            return {
                'bias_probability': bias_score,
                'confidence': result['score'],
                'analysis': f"Bias detection confidence: {result['score']:.2%}"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_clickbait(self, title: str, text: str) -> Dict:
        """Detect clickbait patterns"""
        clickbait_patterns = [
            r'you won\'t believe',
            r'this will shock you',
            r'doctors hate',
            r'one weird trick',
            r'what happens next',
            r'the reason why',
            r'this is why',
            r'number \d+ will',
            r'wait until you see',
            r'this changes everything'
        ]
        
        title_lower = title.lower()
        text_lower = text.lower()
        
        clickbait_indicators = []
        for pattern in clickbait_patterns:
            if re.search(pattern, title_lower) or re.search(pattern, text_lower):
                clickbait_indicators.append(pattern)
        
        # Additional clickbait indicators
        if title.count('!') > 2:
            clickbait_indicators.append('excessive_exclamation')
        
        if len(re.findall(r'\b[A-Z]{2,}\b', title)) > 2:
            clickbait_indicators.append('excessive_caps')
        
        clickbait_score = min(len(clickbait_indicators) * 0.2, 1.0)
        
        return {
            'clickbait_probability': clickbait_score,
            'indicators': clickbait_indicators,
            'analysis': f"Found {len(clickbait_indicators)} clickbait indicators"
        }
    
    def detect_emotional_manipulation(self, text: str) -> Dict:
        """Detect emotional manipulation techniques"""
        emotional_words = {
            'fear': ['terrifying', 'shocking', 'dangerous', 'threat', 'crisis', 'disaster'],
            'anger': ['outrageous', 'disgusting', 'betrayal', 'scandal', 'corrupt'],
            'urgency': ['breaking', 'urgent', 'immediate', 'now', 'today', 'must'],
            'exclusivity': ['exclusive', 'secret', 'hidden', 'revealed', 'exposed']
        }
        
        text_lower = text.lower()
        manipulation_score = 0
        found_techniques = {}
        
        for emotion, words in emotional_words.items():
            count = sum(1 for word in words if word in text_lower)
            if count > 0:
                found_techniques[emotion] = count
                manipulation_score += count * 0.1
        
        return {
            'manipulation_probability': min(manipulation_score, 1.0),
            'techniques_found': found_techniques,
            'analysis': f"Detected {len(found_techniques)} emotional manipulation techniques"
        }
    
    def analyze_citations(self, text: str) -> Dict:
        """Analyze citations and sources in the text"""
        # Look for URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        # Look for citation patterns
        citation_patterns = [
            r'according to',
            r'sources say',
            r'reported by',
            r'study shows',
            r'research indicates',
            r'experts say'
        ]
        
        citations_found = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations_found.extend(matches)
        
        return {
            'url_count': len(urls),
            'citation_phrases': len(citations_found),
            'urls_found': urls[:5],  # Limit to first 5
            'citation_quality_score': min((len(urls) * 0.2 + len(citations_found) * 0.1), 1.0)
        }
    
    def assess_language_quality(self, text: str) -> Dict:
        """Assess language quality and professionalism"""
        # Grammar and spelling indicators (simplified)
        words = text.split()
        
        # Check for common grammar issues
        grammar_issues = 0
        if re.search(r'\b(there|their|they\'re)\b.*\b(there|their|they\'re)\b', text, re.IGNORECASE):
            grammar_issues += 1
        
        if re.search(r'\b(your|you\'re)\b.*\b(your|you\'re)\b', text, re.IGNORECASE):
            grammar_issues += 1
        
        # Check for excessive punctuation
        excessive_punct = len(re.findall(r'[!?]{2,}', text))
        
        # Check for professional language
        professional_indicators = len(re.findall(r'\b(however|therefore|furthermore|moreover|consequently)\b', text, re.IGNORECASE))
        
        quality_score = 1.0
        quality_score -= grammar_issues * 0.1
        quality_score -= excessive_punct * 0.05
        quality_score += professional_indicators * 0.02
        
        return {
            'quality_score': max(0, min(1, quality_score)),
            'grammar_issues': grammar_issues,
            'excessive_punctuation': excessive_punct,
            'professional_language_indicators': professional_indicators
        }

# Cross-reference analyzer
class CrossReferenceAnalyzer:
    def __init__(self):
        self.web_scraper = WebScrapingService()
    
    async def cross_reference_claim(self, original_url: str, claim: str) -> Dict:
        """Cross-reference a claim with other news sources"""
        try:
            # Extract key terms from the claim
            key_terms = self.extract_key_terms(claim)
            
            # Search for similar articles
            similar_articles = await self.search_similar_articles(key_terms)
            
            # Analyze consensus
            consensus_analysis = self.analyze_consensus(similar_articles, claim)
            
            return {
                'original_url': original_url,
                'claim': claim,
                'key_terms': key_terms,
                'similar_articles_found': len(similar_articles),
                'similar_articles': similar_articles[:5],  # Limit results
                'consensus_analysis': consensus_analysis
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def extract_key_terms(self, claim: str) -> List[str]:
        """Extract key terms from a claim for searching"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms[:10]  # Limit to top 10 terms
    
    async def search_similar_articles(self, key_terms: List[str]) -> List[Dict]:
        """Search for similar articles using RSS feeds from reliable sources"""
        similar_articles = []
        
        # RSS feeds from reliable sources
        rss_feeds = [
            'http://feeds.reuters.com/reuters/topNews',
            'https://feeds.npr.org/1001/rss.xml',
            'http://feeds.bbci.co.uk/news/rss.xml',
            'https://rss.cnn.com/rss/edition.rss'
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:  # Limit per feed
                    title = entry.title.lower()
                    summary = entry.get('summary', '').lower()
                    
                    # Check if key terms appear in title or summary
                    matches = sum(1 for term in key_terms if term in title or term in summary)
                    
                    if matches >= 2:  # At least 2 key terms match
                        similar_articles.append({
                            'title': entry.title,
                            'url': entry.link,
                            'published': entry.get('published', ''),
                            'summary': entry.get('summary', ''),
                            'source': feed_url,
                            'key_term_matches': matches
                        })
                        
            except Exception as e:
                print(f"Error processing feed {feed_url}: {e}")
        
        # Sort by number of matching key terms
        similar_articles.sort(key=lambda x: x['key_term_matches'], reverse=True)
        
        return similar_articles
    
    def analyze_consensus(self, similar_articles: List[Dict], original_claim: str) -> Dict:
        """Analyze consensus among similar articles"""
        if not similar_articles:
            return {
                'consensus_score': 0.5,
                'analysis': 'No similar articles found for comparison'
            }
        
        # Simple consensus analysis based on number of similar articles
        consensus_score = min(len(similar_articles) * 0.1, 1.0)
        
        reliable_sources = sum(1 for article in similar_articles 
                             if any(reliable in article['url'] for reliable in RELIABLE_SOURCES.keys()))
        
        if reliable_sources > 0:
            consensus_score += reliable_sources * 0.1
        
        return {
            'consensus_score': min(consensus_score, 1.0),
            'similar_articles_count': len(similar_articles),
            'reliable_sources_count': reliable_sources,
            'analysis': f"Found {len(similar_articles)} similar articles, {reliable_sources} from reliable sources"
        }

# Initialize services
web_scraper = WebScrapingService()
source_analyzer = SourceCredibilityAnalyzer()
content_analyzer = ContentAnalyzer()
cross_reference_analyzer = CrossReferenceAnalyzer()

@app.get("/")
async def root():
    return {"message": "AI News Verification Python API", "status": "running"}

@app.post("/analyze/ml-features", response_model=MLAnalysisResponse)
async def analyze_ml_features(request: ArticleAnalysisRequest):
    """Advanced ML-based analysis of news articles"""
    try:
        text = request.text
        
        # Bias Analysis
        bias_result = ml_models.bias_detector(text)[0]
        bias_analysis = BiasAnalysisResponse(
            bias_score=bias_result['score'] * 100,
            bias_type=bias_result['label'],
            confidence=bias_result['score'],
            explanation=f"Detected {bias_result['label']} with {bias_result['score']:.2%} confidence"
        )
        
        # Sentiment Analysis
        sentiment_result = ml_models.sentiment_analyzer(text)[0]
        emotional_indicators = extract_emotional_indicators(text)
        sentiment_analysis = SentimentAnalysisResponse(
            sentiment=sentiment_result['label'],
            confidence=sentiment_result['score'],
            emotional_indicators=emotional_indicators
        )
        
        # Fake News Probability
        fake_prob = calculate_fake_probability(text)
        
        # Linguistic Features
        linguistic_features = extract_linguistic_features(text)
        
        # Readability Score
        readability = calculate_readability_score(text)
        
        return MLAnalysisResponse(
            bias_analysis=bias_analysis,
            sentiment_analysis=sentiment_analysis,
            fake_probability=fake_prob,
            linguistic_features=linguistic_features,
            readability_score=readability
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/image-verification")
async def verify_image(request: ImageVerificationRequest):
    """Verify images for manipulation and authenticity"""
    try:
        # Download image
        response = requests.get(request.image_url)
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Perform image analysis
        manipulation_score = detect_image_manipulation(opencv_image)
        metadata_analysis = analyze_image_metadata(response.content)
        reverse_search_results = perform_reverse_image_search(request.image_url)
        
        return {
            "manipulation_probability": manipulation_score,
            "metadata_analysis": metadata_analysis,
            "reverse_search_results": reverse_search_results,
            "authenticity_score": calculate_image_authenticity_score(
                manipulation_score, metadata_analysis, reverse_search_results
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image verification failed: {str(e)}")

@app.post("/train/update-model")
async def update_ml_model(request: MLTrainingRequest):
    """Update ML models with new training data"""
    try:
        # Extract features from articles
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        features = vectorizer.fit_transform([article['text'] for article in request.articles])
        
        # Train classifier
        classifier = LogisticRegression()
        classifier.fit(features, request.labels)
        
        # Save updated model
        os.makedirs("models", exist_ok=True)
        joblib.dump(classifier, "models/fake_news_classifier.joblib")
        joblib.dump(vectorizer, "models/vectorizer.joblib")
        
        # Update global model
        ml_models.fake_news_classifier = classifier
        
        return {
            "status": "success",
            "message": f"Model updated with {len(request.articles)} new samples",
            "accuracy": "Training accuracy will be calculated in production"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/analyze/batch-processing")
async def batch_process_articles(articles: List[str]):
    """Process multiple articles efficiently using Python's multiprocessing"""
    try:
        import multiprocessing as mp
        from functools import partial
        
        # Create partial function with loaded models
        analyze_func = partial(analyze_single_article_ml, ml_models)
        
        # Use multiprocessing for batch analysis
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(analyze_func, articles)
        
        return {
            "results": results,
            "total_processed": len(results),
            "processing_time": "Calculated in production"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/analyze/url")
async def analyze_url(request: URLAnalysisRequest):
    """Comprehensive analysis of a news article from URL"""
    try:
        url = request.url
        
        # Extract article content
        print(f"Extracting content from: {url}")
        content = await web_scraper.extract_article_content(url)
        
        if 'error' in content:
            raise HTTPException(status_code=400, detail=content['error'])
        
        # Analyze source credibility
        print("Analyzing source credibility...")
        source_analysis = source_analyzer.analyze_domain_credibility(url)
        
        # Analyze content
        print("Analyzing content...")
        content_analysis = content_analyzer.analyze_article_content(content)
        
        # Cross-reference if deep analysis requested
        cross_reference = None
        if request.deep_analysis and content.get('text'):
            print("Cross-referencing claims...")
            # Use first paragraph as main claim for cross-referencing
            first_paragraph = content['text'].split('\n')[0][:200]
            cross_reference = await cross_reference_analyzer.cross_reference_claim(url, first_paragraph)
        
        # Calculate overall scores
        overall_analysis = calculate_overall_scores(source_analysis, content_analysis, cross_reference)
        
        return {
            'url': url,
            'extracted_content': {
                'title': content.get('title', ''),
                'text_preview': content.get('text', '')[:500] + '...' if content.get('text') else '',
                'word_count': len(content.get('text', '').split()),
                'extraction_method': content.get('extraction_method', 'unknown')
            },
            'source_analysis': source_analysis,
            'content_analysis': content_analysis,
            'cross_reference': cross_reference,
            'overall_analysis': overall_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch-urls")
async def analyze_batch_urls(request: BatchURLRequest):
    """Analyze multiple URLs in batch"""
    try:
        results = []
        
        for url in request.urls[:10]:  # Limit to 10 URLs
            try:
                # Create individual request
                individual_request = URLAnalysisRequest(url=url, deep_analysis=False)
                result = await analyze_url(individual_request)
                results.append(result)
                
                # Add delay to avoid overwhelming servers
                await asyncio.sleep(1)
                
            except Exception as e:
                results.append({
                    'url': url,
                    'error': str(e),
                    'analysis_timestamp': datetime.now().isoformat()
                })
        
        # Calculate batch statistics
        successful_analyses = [r for r in results if 'error' not in r]
        batch_stats = calculate_batch_statistics(successful_analyses)
        
        return {
            'results': results,
            'batch_statistics': batch_stats,
            'total_analyzed': len(results),
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(results) - len(successful_analyses)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/analyze/source")
async def analyze_source(request: SourceAnalysisRequest):
    """Analyze a news source domain for credibility"""
    try:
        domain = request.domain.replace('http://', '').replace('https://', '').replace('www.', '')
        
        analysis = source_analyzer.analyze_domain_credibility(f"https://{domain}")
        
        return {
            'domain': domain,
            'analysis': analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Source analysis failed: {str(e)}")

@app.post("/cross-reference")
async def cross_reference_claim(request: CrossReferenceRequest):
    """Cross-reference a specific claim with other sources"""
    try:
        result = await cross_reference_analyzer.cross_reference_claim(
            request.article_url, 
            request.claim
        )
        
        return {
            'cross_reference_result': result,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-reference failed: {str(e)}")

# Helper functions
def extract_emotional_indicators(text: str) -> List[str]:
    """Extract emotional indicators from text"""
    emotional_words = {
        'anger': ['angry', 'furious', 'outraged', 'livid', 'enraged'],
        'fear': ['afraid', 'terrified', 'scared', 'frightened', 'worried'],
        'joy': ['happy', 'excited', 'thrilled', 'delighted', 'ecstatic'],
        'sadness': ['sad', 'depressed', 'miserable', 'heartbroken', 'devastated']
    }
    
    indicators = []
    text_lower = text.lower()
    
    for emotion, words in emotional_words.items():
        for word in words:
            if word in text_lower:
                indicators.append(f"{emotion}: {word}")
    
    return indicators

def calculate_fake_probability(text: str) -> float:
    """Calculate probability of fake news using ML model"""
    try:
        if ml_models.fake_news_classifier is None:
            # Fallback to simple heuristics
            suspicious_phrases = ['breaking:', 'you won\'t believe', 'shocking', 'exclusive']
            score = sum(1 for phrase in suspicious_phrases if phrase in text.lower())
            return min(score * 0.25, 1.0)
        
        # Use trained model (placeholder implementation)
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        features = vectorizer.fit_transform([text])
        probability = ml_models.fake_news_classifier.predict_proba(features)[0][1]
        return float(probability)
        
    except Exception:
        return 0.5  # Default uncertainty

def extract_linguistic_features(text: str) -> dict:
    """Extract linguistic features from text"""
    words = text.split()
    sentences = text.split('.')
    
    return {
        "avg_word_length": np.mean([len(word) for word in words]),
        "avg_sentence_length": np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()]),
        "exclamation_ratio": text.count('!') / len(text) if len(text) > 0 else 0,
        "question_ratio": text.count('?') / len(text) if len(text) > 0 else 0,
        "caps_ratio": sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        "unique_word_ratio": len(set(words)) / len(words) if len(words) > 0 else 0
    }

def calculate_readability_score(text: str) -> float:
    """Calculate Flesch Reading Ease score"""
    words = len(text.split())
    sentences = len([s for s in text.split('.') if s.strip()])
    syllables = sum(count_syllables(word) for word in text.split())
    
    if sentences == 0 or words == 0:
        return 0
    
    score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
    return max(0, min(100, score))

def count_syllables(word: str) -> int:
    """Count syllables in a word (simplified)"""
    word = word.lower()
    vowels = 'aeiouy'
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    if word.endswith('e'):
        syllable_count -= 1
    
    return max(1, syllable_count)

def detect_image_manipulation(image: np.ndarray) -> float:
    """Detect image manipulation using computer vision techniques"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density (manipulated images often have irregular edges)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Simple heuristic: higher edge density might indicate manipulation
        manipulation_score = min(edge_density * 2, 1.0)
        
        return float(manipulation_score)
        
    except Exception:
        return 0.5  # Default uncertainty

def analyze_image_metadata(image_data: bytes) -> dict:
    """Analyze image metadata for authenticity indicators"""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        image = Image.open(io.BytesIO(image_data))
        exifdata = image.getexif()
        
        metadata = {}
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            metadata[tag] = str(data)
        
        # Analyze metadata for authenticity indicators
        authenticity_indicators = {
            "has_camera_info": "Make" in metadata or "Model" in metadata,
            "has_gps_info": "GPSInfo" in metadata,
            "has_timestamp": "DateTime" in metadata,
            "software_modified": "Software" in metadata
        }
        
        return {
            "metadata": metadata,
            "authenticity_indicators": authenticity_indicators
        }
        
    except Exception:
        return {"error": "Could not analyze metadata"}

def perform_reverse_image_search(image_url: str) -> dict:
    """Perform reverse image search (placeholder implementation)"""
    # In production, integrate with Google Images API or TinEye
    return {
        "similar_images_found": 0,
        "earliest_appearance": None,
        "search_status": "placeholder_implementation"
    }

def calculate_image_authenticity_score(manipulation_score: float, metadata: dict, reverse_search: dict) -> float:
    """Calculate overall image authenticity score"""
    score = 1.0
    
    # Reduce score based on manipulation probability
    score -= manipulation_score * 0.4
    
    # Adjust based on metadata
    if "authenticity_indicators" in metadata:
        indicators = metadata["authenticity_indicators"]
        if not indicators.get("has_camera_info", False):
            score -= 0.2
        if indicators.get("software_modified", False):
            score -= 0.3
    
    return max(0, min(1, score))

def analyze_single_article_ml(models, article_text: str) -> dict:
    """Analyze single article with ML models (for batch processing)"""
    try:
        # Simplified analysis for batch processing
        sentiment = models.sentiment_analyzer(article_text[:512])[0]  # Limit text length
        bias = models.bias_detector(article_text[:512])[0]
        
        return {
            "sentiment": sentiment,
            "bias": bias,
            "fake_probability": calculate_fake_probability(article_text),
            "readability": calculate_readability_score(article_text)
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_overall_scores(source_analysis: Dict, content_analysis: Dict, cross_reference: Dict = None) -> Dict:
    """Calculate overall authenticity and credibility scores"""
    
    # Source credibility score (40% weight)
    source_score = source_analysis.get('credibility_score', 50) / 100
    
    # Content quality score (40% weight)
    content_score = 0.5  # Default neutral
    if 'language_quality' in content_analysis:
        content_score = content_analysis['language_quality'].get('quality_score', 0.5)
    
    # Bias penalty
    if 'bias' in content_analysis:
        bias_prob = content_analysis['bias'].get('bias_probability', 0)
        content_score *= (1 - bias_prob * 0.5)  # Reduce score based on bias
    
    # Clickbait penalty
    if 'clickbait' in content_analysis:
        clickbait_prob = content_analysis['clickbait'].get('clickbait_probability', 0)
        content_score *= (1 - clickbait_prob * 0.3)
    
    # Cross-reference score (20% weight)
    cross_ref_score = 0.5  # Default neutral
    if cross_reference and 'consensus_analysis' in cross_reference:
        cross_ref_score = cross_reference['consensus_analysis'].get('consensus_score', 0.5)
    
    # Calculate weighted overall score
    overall_score = (source_score * 0.4 + content_score * 0.4 + cross_ref_score * 0.2)
    
    # Determine verdict
    if overall_score >= 0.7:
        verdict = "LIKELY_AUTHENTIC"
    elif overall_score >= 0.4:
        verdict = "UNCERTAIN"
    else:
        verdict = "LIKELY_FAKE"
    
    return {
        'overall_authenticity_score': round(overall_score * 100, 2),
        'source_credibility_score': round(source_score * 100, 2),
        'content_quality_score': round(content_score * 100, 2),
        'cross_reference_score': round(cross_ref_score * 100, 2),
        'verdict': verdict,
        'confidence': min(90, max(60, overall_score * 100)),  # Confidence between 60-90%
        'scoring_breakdown': {
            'source_weight': '40%',
            'content_weight': '40%',
            'cross_reference_weight': '20%'
        }
    }

def calculate_batch_statistics(results: List[Dict]) -> Dict:
    """Calculate statistics for batch analysis"""
    if not results:
        return {}
    
    verdicts = [r['overall_analysis']['verdict'] for r in results if 'overall_analysis' in r]
    scores = [r['overall_analysis']['overall_authenticity_score'] for r in results if 'overall_analysis' in r]
    
    return {
        'average_authenticity_score': round(np.mean(scores), 2) if scores else 0,
        'verdict_distribution': {
            'LIKELY_AUTHENTIC': verdicts.count('LIKELY_AUTHENTIC'),
            'UNCERTAIN': verdicts.count('UNCERTAIN'),
            'LIKELY_FAKE': verdicts.count('LIKELY_FAKE')
        },
        'total_analyzed': len(results)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
