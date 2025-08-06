import { generateObject } from "ai"
import { openai } from "@ai-sdk/openai"
import { z } from "zod"
import type { NextRequest } from "next/server"
import { createHash } from "crypto"

const batchVerificationSchema = z.object({
  results: z.array(
    z.object({
      id: z.string(),
      overallScore: z.number().min(0).max(100),
      credibilityScore: z.number().min(0).max(100),
      biasScore: z.number().min(0).max(100),
      factualityScore: z.number().min(0).max(100),
      sourceReliability: z.number().min(0).max(100),
      confidence: z.number().min(0).max(100),
      verdict: z.enum(["AUTHENTIC", "SUSPICIOUS", "LIKELY_FAKE"]),
      analysis: z.object({
        strengths: z.array(z.string()),
        concerns: z.array(z.string()),
        recommendations: z.array(z.string()),
        redFlags: z.array(z.string()),
      }),
      details: z.object({
        languageAnalysis: z.string(),
        sourceAnalysis: z.string(),
        contentAnalysis: z.string(),
        factChecking: z.string(),
        temporalAnalysis: z.string(),
        citationAnalysis: z.string(),
      }),
      metadata: z.object({
        wordCount: z.number(),
        readingLevel: z.string(),
        emotionalTone: z.string(),
        politicalLean: z.string(),
      }),
    }),
  ),
  summary: z.object({
    totalAnalyzed: z.number(),
    authentic: z.number(),
    suspicious: z.number(),
    likelyFake: z.number(),
    averageScore: z.number(),
  }),
})

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const { articles } = await request.json()

    if (!articles || !Array.isArray(articles) || articles.length === 0) {
      return Response.json({ error: "Articles array is required" }, { status: 400 })
    }

    if (articles.length > 10) {
      return Response.json({ error: "Maximum 10 articles per batch" }, { status: 400 })
    }

    // Prepare articles for analysis
    const articlesText = articles
      .map(
        (article, index) =>
          `ARTICLE ${index + 1}:
${article.text}

---`,
      )
      .join("\n")

    const { object } = await generateObject({
      model: openai("gpt-4o"),
      schema: batchVerificationSchema,
      system: `You are an expert fact-checker performing batch analysis of multiple news articles. Analyze each article independently using the same comprehensive verification framework:

ANALYSIS FRAMEWORK:
1. Language Analysis: Emotional manipulation, sensationalism, writing quality
2. Source Credibility: Domain reputation, author credentials, editorial standards
3. Factual Verification: Evidence quality, citations, verifiable claims
4. Bias Detection: Political/ideological bias, one-sided reporting
5. Temporal Analysis: Timeline consistency, chronological accuracy
6. Citation Analysis: Source quality and transparency

SCORING (0-100%):
- Overall Score: Weighted composite of all factors
- Credibility: Professional standards and expertise
- Bias: Higher = more biased (0 = neutral)
- Factuality: Evidence quality and accuracy
- Source Reliability: Publication credibility

VERDICTS:
- AUTHENTIC (70-100%): High credibility, factual, minimal bias
- SUSPICIOUS (40-69%): Mixed signals, needs verification
- LIKELY_FAKE (0-39%): Multiple red flags, unreliable

For each article, provide:
- Unique analysis ID
- Complete scoring breakdown
- Detailed analysis in all categories
- Specific strengths, concerns, red flags, and recommendations
- Metadata assessment

Generate summary statistics for the batch.`,
      prompt: `Analyze these ${articles.length} news articles for authenticity and credibility:

${articlesText}

Provide comprehensive individual analysis for each article plus batch summary statistics.`,
    })

    // Add computed data
    const results = object.results.map((result, index) => {
      const articleHash = createHash("md5").update(articles[index].text).digest("hex")
      const wordCount = articles[index].text.split(/\s+/).length

      return {
        ...result,
        id: `batch_${articleHash}_${Date.now()}_${index}`,
        metadata: {
          ...result.metadata,
          wordCount,
          analysisTime: Date.now() - startTime,
        },
      }
    })

    const finalResult = {
      results,
      summary: object.summary,
    }

    console.log(`Batch analysis completed: ${articles.length} articles, Average score: ${object.summary.averageScore}%`)

    return Response.json(finalResult)
  } catch (error) {
    console.error("Error in batch verification:", error)
    return Response.json({ error: "Failed to verify articles" }, { status: 500 })
  }
}
