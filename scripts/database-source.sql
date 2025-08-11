-- News Verification System Database Schema
-- Run this script to set up the database tables

-- Create database (if using PostgreSQL)
-- CREATE DATABASE news_verification;

-- Source reliability tracking table
CREATE TABLE IF NOT EXISTS source_reliability (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(255) UNIQUE NOT NULL,
    reliability_score INTEGER NOT NULL DEFAULT 50,
    total_articles INTEGER NOT NULL DEFAULT 0,
    accurate_articles INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Article analysis results table
CREATE TABLE IF NOT EXISTS article_analysis (
    id VARCHAR(255) PRIMARY KEY,
    article_hash VARCHAR(64) NOT NULL,
    source_url TEXT,
    source_domain VARCHAR(255),
    overall_score INTEGER NOT NULL,
    credibility_score INTEGER NOT NULL,
    bias_score INTEGER NOT NULL,
    factuality_score INTEGER NOT NULL,
    source_reliability_score INTEGER NOT NULL,
    confidence_score INTEGER NOT NULL,
    verdict VARCHAR(20) NOT NULL,
    word_count INTEGER,
    reading_level VARCHAR(50),
    emotional_tone VARCHAR(50),
    political_lean VARCHAR(50),
    analysis_time INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback table for continuous learning
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    analysis_id VARCHAR(255) REFERENCES article_analysis(id),
    user_verdict VARCHAR(20),
    confidence INTEGER,
    feedback_text TEXT,
    helpful_rating INTEGER CHECK (helpful_rating >= 1 AND helpful_rating <= 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Batch analysis tracking
CREATE TABLE IF NOT EXISTS batch_analysis (
    id VARCHAR(255) PRIMARY KEY,
    total_articles INTEGER NOT NULL,
    authentic_count INTEGER NOT NULL,
    suspicious_count INTEGER NOT NULL,
    fake_count INTEGER NOT NULL,
    average_score DECIMAL(5,2),
    analysis_time INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Known fake news domains (blacklist)
CREATE TABLE IF NOT EXISTS blacklisted_domains (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(255) UNIQUE NOT NULL,
    reason TEXT,
    severity VARCHAR(20) DEFAULT 'HIGH',
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified_by VARCHAR(100)
);

-- Trusted news sources (whitelist)
CREATE TABLE IF NOT EXISTS trusted_sources (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(255) UNIQUE NOT NULL,
    organization_name VARCHAR(255),
    credibility_rating INTEGER DEFAULT 90,
    editorial_standards TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified_by VARCHAR(100)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_article_analysis_hash ON article_analysis(article_hash);
CREATE INDEX IF NOT EXISTS idx_article_analysis_domain ON article_analysis(source_domain);
CREATE INDEX IF NOT EXISTS idx_article_analysis_verdict ON article_analysis(verdict);
CREATE INDEX IF NOT EXISTS idx_article_analysis_created ON article_analysis(created_at);
CREATE INDEX IF NOT EXISTS idx_source_reliability_domain ON source_reliability(domain);
CREATE INDEX IF NOT EXISTS idx_user_feedback_analysis ON user_feedback(analysis_id);

-- Insert some initial trusted sources
INSERT INTO trusted_sources (domain, organization_name, credibility_rating, editorial_standards) VALUES
('reuters.com', 'Reuters', 95, 'Strict editorial guidelines, fact-checking protocols'),
('ap.org', 'Associated Press', 95, 'Rigorous fact-checking, neutral reporting standards'),
('bbc.com', 'BBC News', 90, 'Editorial guidelines, impartiality requirements'),
('npr.org', 'National Public Radio', 88, 'Editorial standards, transparency in sourcing'),
('pbs.org', 'PBS NewsHour', 87, 'Journalistic integrity standards'),
('nytimes.com', 'The New York Times', 85, 'Editorial standards, fact-checking department'),
('washingtonpost.com', 'The Washington Post', 85, 'Editorial policies, verification processes'),
('wsj.com', 'The Wall Street Journal', 84, 'Editorial standards, business journalism ethics'),
('theguardian.com', 'The Guardian', 82, 'Editorial code, fact-checking procedures'),
('cnn.com', 'CNN', 78, 'Editorial standards, verification protocols')
ON CONFLICT (domain) DO NOTHING;

-- Insert some known problematic domains (examples - update with current data)
INSERT INTO blacklisted_domains (domain, reason, severity) VALUES
('fakenews.com', 'Known disinformation site', 'HIGH'),
('conspiracy-news.net', 'Promotes conspiracy theories', 'HIGH'),
('clickbait-central.com', 'Misleading headlines, poor fact-checking', 'MEDIUM')
ON CONFLICT (domain) DO NOTHING;

-- Create a view for analysis statistics
CREATE OR REPLACE VIEW analysis_statistics AS
SELECT 
    DATE(created_at) as analysis_date,
    COUNT(*) as total_analyses,
    AVG(overall_score) as avg_score,
    COUNT(CASE WHEN verdict = 'AUTHENTIC' THEN 1 END) as authentic_count,
    COUNT(CASE WHEN verdict = 'SUSPICIOUS' THEN 1 END) as suspicious_count,
    COUNT(CASE WHEN verdict = 'LIKELY_FAKE' THEN 1 END) as fake_count
FROM article_analysis 
GROUP BY DATE(created_at)
ORDER BY analysis_date DESC;

-- Create a function to update source reliability based on analysis results
CREATE OR REPLACE FUNCTION update_source_reliability()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO source_reliability (domain, reliability_score, total_articles, accurate_articles)
    VALUES (
        NEW.source_domain,
        NEW.overall_score,
        1,
        CASE WHEN NEW.overall_score >= 70 THEN 1 ELSE 0 END
    )
    ON CONFLICT (domain) DO UPDATE SET
        total_articles = source_reliability.total_articles + 1,
        accurate_articles = source_reliability.accurate_articles + 
            CASE WHEN NEW.overall_score >= 70 THEN 1 ELSE 0 END,
        reliability_score = ROUND(
            (source_reliability.accurate_articles::FLOAT / source_reliability.total_articles::FLOAT) * 100
        ),
        last_updated = CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update source reliability
CREATE TRIGGER update_source_reliability_trigger
    AFTER INSERT ON article_analysis
    FOR EACH ROW
    WHEN (NEW.source_domain IS NOT NULL)
    EXECUTE FUNCTION update_source_reliability();
