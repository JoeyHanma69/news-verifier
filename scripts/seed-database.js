// Database seeding script for News Verification System
// Run with: node scripts/seed-database.js

const { Pool } = require("pg")

// Database configuration
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || "postgresql://localhost:5432/news_verification",
})

async function seedDatabase() {
  const client = await pool.connect()

  try {
    console.log("Starting database seeding...")

    // Seed additional trusted sources
    const trustedSources = [
      { domain: "politico.com", name: "Politico", rating: 82 },
      { domain: "axios.com", name: "Axios", rating: 80 },
      { domain: "thehill.com", name: "The Hill", rating: 78 },
      { domain: "usatoday.com", name: "USA Today", rating: 76 },
      { domain: "latimes.com", name: "Los Angeles Times", rating: 79 },
      { domain: "chicagotribune.com", name: "Chicago Tribune", rating: 77 },
      { domain: "seattletimes.com", name: "The Seattle Times", rating: 75 },
      { domain: "denverpost.com", name: "The Denver Post", rating: 74 },
    ]

    for (const source of trustedSources) {
      await client.query(
        `
        INSERT INTO trusted_sources (domain, organization_name, credibility_rating, editorial_standards)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (domain) DO NOTHING
      `,
        [source.domain, source.name, source.rating, "Standard editorial guidelines"],
      )
    }

    // Seed sample analysis data for testing
    const sampleAnalyses = [
      {
        id: "sample_001",
        hash: "abc123",
        url: "https://reuters.com/sample-article",
        domain: "reuters.com",
        overall: 92,
        credibility: 95,
        bias: 15,
        factuality: 90,
        source_rel: 95,
        confidence: 88,
        verdict: "AUTHENTIC",
        word_count: 450,
        reading_level: "College",
        tone: "Neutral",
        lean: "Neutral",
      },
      {
        id: "sample_002",
        hash: "def456",
        url: "https://suspicious-news.com/article",
        domain: "suspicious-news.com",
        overall: 35,
        credibility: 25,
        bias: 75,
        factuality: 40,
        source_rel: 30,
        confidence: 82,
        verdict: "LIKELY_FAKE",
        word_count: 280,
        reading_level: "High School",
        tone: "Angry",
        lean: "Extreme",
      },
    ]

    for (const analysis of sampleAnalyses) {
      await client.query(
        `
        INSERT INTO article_analysis (
          id, article_hash, source_url, source_domain, overall_score,
          credibility_score, bias_score, factuality_score, source_reliability_score,
          confidence_score, verdict, word_count, reading_level, emotional_tone,
          political_lean, analysis_time
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        ON CONFLICT (id) DO NOTHING
      `,
        [
          analysis.id,
          analysis.hash,
          analysis.url,
          analysis.domain,
          analysis.overall,
          analysis.credibility,
          analysis.bias,
          analysis.factuality,
          analysis.source_rel,
          analysis.confidence,
          analysis.verdict,
          analysis.word_count,
          analysis.reading_level,
          analysis.tone,
          analysis.lean,
          1500,
        ],
      )
    }

    // Initialize source reliability scores
    await client.query(`
      INSERT INTO source_reliability (domain, reliability_score, total_articles, accurate_articles)
      SELECT domain, credibility_rating, 1, 1
      FROM trusted_sources
      ON CONFLICT (domain) DO NOTHING
    `)

    console.log("Database seeding completed successfully!")

    // Display statistics
    const stats = await client.query(`
      SELECT 
        (SELECT COUNT(*) FROM trusted_sources) as trusted_sources,
        (SELECT COUNT(*) FROM blacklisted_domains) as blacklisted_domains,
        (SELECT COUNT(*) FROM article_analysis) as sample_analyses,
        (SELECT COUNT(*) FROM source_reliability) as source_reliability_entries
    `)

    console.log("Seeding Statistics:", stats.rows[0])
  } catch (error) {
    console.error("Error seeding database:", error)
  } finally {
    client.release()
    await pool.end()
  }
}

// Run seeding if this file is executed directly
if (require.main === module) {
  seedDatabase().catch(console.error)
}

module.exports = { seedDatabase }
