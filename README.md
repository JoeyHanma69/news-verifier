# AI News Verification System

A comprehensive AI-powered system for detecting fake news and verifying article authenticity using advanced natural language processing, source credibility analysis, and fact-checking techniques.

## Features

### Core Functionality

- **Single Article Analysis**: Comprehensive verification of individual news articles
- **URL Analysis**: Automatic content extraction and verification from news URLs
- **Batch Processing**: Analyze multiple articles simultaneously
- **Real-time Verification**: Instant analysis with detailed scoring

### Advanced Analysis

- **Multi-factor Scoring**: Credibility, bias, factuality, and source reliability
- **Language Analysis**: Detect emotional manipulation and sensationalism
- **Source Credibility**: Domain reputation and editorial standards assessment
- **Bias Detection**: Political and ideological bias identification
- **Temporal Analysis**: Timeline consistency and chronological accuracy
- **Citation Analysis**: Source quality and transparency evaluation

### Technical Features

- **Database Integration**: PostgreSQL for source tracking and analysis history
- **Caching System**: Redis for performance optimization
- **Rate Limiting**: API protection and usage management
- **Responsive Design**: Mobile-friendly interface
- **Real-time Updates**: Live analysis progress and results

## Quick Start

### Prerequisites

- Node.js 18+
- PostgreSQL 13+
- OpenAI API key
- Redis (optional, for caching)

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/your-username/ai-news-verification.git
   cd ai-news-verification
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   npm install
   \`\`\`

3. **Set up environment variables**
   \`\`\`bash
   cp .env.example .env.local

   # Edit .env.local with your configuration

   \`\`\`

4. **Set up the database**
   \`\`\`bash

   # Create PostgreSQL database

   createdb news_verification

   # Run database setup

   npm run db:setup

   # Seed with initial data

   npm run db:seed
   \`\`\`

5. **Start the development server**
   \`\`\`bash
   npm run dev
   \`\`\`

6. **Open your browser**
   Navigate to `http://localhost:3000`

## Detailed Setup Instructions

### Step 1: Environment Configuration

Create a `.env.local` file with the following variables:

\`\`\`env
OPENAI_API_KEY=sk-your-openai-api-key
DATABASE_URL=postgresql://username:password@localhost:5432/news_verification
REDIS_URL=redis://localhost:6379
NODE_ENV=development
\`\`\`

### Step 2: Database Setup

1. **Install PostgreSQL**
   \`\`\`bash

   # macOS

   brew install postgresql
   brew services start postgresql

   # Ubuntu

   sudo apt-get install postgresql postgresql-contrib
   sudo systemctl start postgresql

   # Windows

   # Download from https://www.postgresql.org/download/windows/

   \`\`\`

2. **Create database and user**
   \`\`\`sql
   sudo -u postgres psql
   CREATE DATABASE news_verification;
   CREATE USER news_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE news_verification TO news_user;
   \q
   \`\`\`

3. **Run database migrations**
   \`\`\`bash
   npm run db:setup
   npm run db:seed
   \`\`\`

### Step 3: OpenAI API Setup

1. **Get API Key**

   - Visit [OpenAI Platform](https://platform.openai.com/)
   - Create an account or sign in
   - Navigate to API Keys section
   - Create a new API key

2. **Configure API Key**
   \`\`\`bash
   echo "OPENAI_API_KEY=sk-your-key-here" >> .env.local
   \`\`\`

### Step 4: Optional Redis Setup (for caching)

1. **Install Redis**
   \`\`\`bash

   # macOS

   brew install redis
   brew services start redis

   # Ubuntu

   sudo apt-get install redis-server
   sudo systemctl start redis

   # Windows

   # Download from https://redis.io/download

   \`\`\`

2. **Configure Redis**
   \`\`\`bash
   echo "REDIS_URL=redis://localhost:6379" >> .env.local
   \`\`\`

## Usage Guide

### Single Article Analysis

1. **Navigate to the Single Article tab**
2. **Paste article text** into the text area
3. **Add source URL** (optional but recommended)
4. **Click "Verify Article"**
5. **Review comprehensive results** including:
   - Overall authenticity score
   - Detailed breakdown (credibility, bias, factuality, source reliability)
   - Specific strengths and concerns
   - Actionable recommendations

### URL Analysis

1. **Navigate to the From URL tab**
2. **Enter the news article URL**
3. **Click "Analyze URL"**
4. **System automatically extracts content** and performs analysis
5. **Review results** with extracted text and verification scores

### Batch Analysis

1. **Navigate to the Batch Analysis tab**
2. **Enter multiple articles** separated by `---`
3. **Click "Analyze Batch"**
4. **Review batch summary** and individual article results
5. **Export results** for further analysis

## API Endpoints

### POST /api/verify-news

Analyze a single news article

**Request Body:**
\`\`\`json
{
"articleText": "Article content here...",
"sourceUrl": "https://example.com/article",
"analysisType": "comprehensive"
}
\`\`\`

**Response:**
\`\`\`json
{
"id": "analysis_abc123_1234567890",
"overallScore": 85,
"credibilityScore": 90,
"biasScore": 20,
"factualityScore": 88,
"sourceReliability": 92,
"confidence": 87,
"verdict": "AUTHENTIC",
"analysis": {
"strengths": ["Well-sourced", "Balanced reporting"],
"concerns": ["Minor bias detected"],
"recommendations": ["Cross-reference with other sources"],
"redFlags": []
},
"details": {
"languageAnalysis": "Professional writing style...",
"sourceAnalysis": "Reputable publication...",
"contentAnalysis": "Factual claims supported...",
"factChecking": "Claims verified against...",
"temporalAnalysis": "Timeline consistent...",
"citationAnalysis": "Quality sources cited..."
},
"metadata": {
"wordCount": 450,
"readingLevel": "College",
"emotionalTone": "Neutral",
"politicalLean": "Neutral",
"analysisTime": 2500
}
}
\`\`\`

### POST /api/verify-batch

Analyze multiple articles

**Request Body:**
\`\`\`json
{
"articles": [
{"text": "Article 1 content..."},
{"text": "Article 2 content..."}
]
}
\`\`\`

### POST /api/verify-url

Analyze article from URL

**Request Body:**
\`\`\`json
{
"url": "https://example.com/news-article"
}
\`\`\`

## Advanced Features

### Source Reliability Tracking

The system maintains a database of source reliability scores based on historical analysis:

\`\`\`sql
-- View source reliability statistics
SELECT domain, reliability_score, total_articles, accurate_articles
FROM source_reliability
ORDER BY reliability_score DESC;
\`\`\`

### Continuous Learning

User feedback is collected to improve accuracy:

\`\`\`sql
-- Add user feedback
INSERT INTO user_feedback (analysis_id, user_verdict, confidence, feedback_text)
VALUES ('analysis_123', 'AUTHENTIC', 90, 'This analysis was accurate');
\`\`\`

### Performance Monitoring

Track system performance and accuracy:

\`\`\`sql
-- View analysis statistics
SELECT \* FROM analysis_statistics
WHERE analysis_date >= CURRENT_DATE - INTERVAL '7 days';
\`\`\`

## Deployment

### Production Deployment on Vercel

1. **Push to GitHub**
   \`\`\`bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   \`\`\`

2. **Deploy to Vercel**
   \`\`\`bash
   npm install -g vercel
   vercel --prod
   \`\`\`

3. **Configure Environment Variables**
   - Add all environment variables in Vercel dashboard
   - Set up production database (Neon, Supabase, or AWS RDS)

### Docker Deployment

1. **Create Dockerfile**
   \`\`\`dockerfile
   FROM node:18-alpine
   WORKDIR /app
   COPY package\*.json ./
   RUN npm ci --only=production
   COPY . .
   RUN npm run build
   EXPOSE 3000
   CMD ["npm", "start"]
   \`\`\`

2. **Build and run**
   \`\`\`bash
   docker build -t news-verifier .
   docker run -p 3000:3000 --env-file .env news-verifier
   \`\`\`

## Security Considerations

### API Security

- Rate limiting implemented (60 requests/minute by default)
- Input validation and sanitization
- SQL injection prevention
- XSS protection

### Data Privacy

- No article content stored permanently
- Analysis results can be anonymized
- GDPR compliance considerations
- Secure API key management

## Performance Optimization

### Caching Strategy

- Redis caching for repeated analyses
- Database query optimization
- CDN for static assets
- Response compression

### Scaling Considerations

- Horizontal scaling with load balancers
- Database read replicas
- Queue system for batch processing
- Microservices architecture for high volume

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   \`\`\`bash

   # Check PostgreSQL status

   sudo systemctl status postgresql

   # Verify connection string

   psql $DATABASE_URL -c "SELECT 1;"
   \`\`\`

2. **OpenAI API Errors**
   \`\`\`bash

   # Test API key

   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
    https://api.openai.com/v1/models
   \`\`\`

3. **Memory Issues**
   \`\`\`bash
   # Increase Node.js memory limit
   NODE_OPTIONS="--max-old-space-size=4096" npm run dev
   \`\`\`

### Performance Issues

1. **Slow Analysis**

   - Enable Redis caching
   - Optimize database queries
   - Use connection pooling

2. **High Memory Usage**
   - Implement streaming for large articles
   - Add garbage collection optimization
   - Use worker threads for CPU-intensive tasks

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
   \`\`\`bash
   git checkout -b feature/new-analysis-method
   \`\`\`

3. **Make changes and test**
   \`\`\`bash
   npm run test
   npm run lint
   \`\`\`

4. **Submit pull request**

### Code Standards

- TypeScript for type safety
- ESLint for code quality
- Prettier for formatting
- Jest for testing

## License

MIT License - see LICENSE file for details

## Support

- **Documentation**: [docs.newsverifier.com](https://docs.newsverifier.com)
- **Issues**: [GitHub Issues](https://github.com/your-username/ai-news-verification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ai-news-verification/discussions)
- **Email**: support@newsverifier.com

## Roadmap

### Version 2.0 Features

- [ ] Image and video verification
- [ ] Real-time fact-checking API integration
- [ ] Browser extension
- [ ] Mobile app
- [ ] Multi-language support
- [ ] Advanced ML model training
- [ ] Social media integration
- [ ] Collaborative fact-checking

### Version 3.0 Features

- [ ] Blockchain verification
- [ ] Decentralized fact-checking network
- [ ] AI model marketplace
- [ ] Enterprise dashboard
- [ ] White-label solutions
