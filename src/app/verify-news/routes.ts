import { generateObject } from "ai"
import { openai } from "@ai-sdk/openai"
import { z } from "zod"
import type { NextRequest } from "next/server"
import { createHash } from "crypto"

const verificationSchema = z.object({
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
})

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const { articleText, sourceUrl, analysisType = "comprehensive" } = await request.json()

    if (!articleText) {
      return Response.json({ error: "Article text is required" }, { status: 400 })
    }

    // Generate unique ID for this analysis
    const articleHash = createHash("md5").update(articleText).digest("hex")
    const analysisId = `analysis_${articleHash}_${Date.now()}`

    // Enhanced system prompt for comprehensive analysis
    const systemPrompt = `You are an expert fact-checker and news verification specialist with advanced training in journalism, linguistics, and misinformation detection. Analyze news articles for authenticity, bias, and credibility using multiple sophisticated verification techniques.

COMPREHENSIVE ANALYSIS FRAMEWORK:

1. LANGUAGE ANALYSIS:
   - Detect emotional manipulation, sensationalism, clickbait patterns
   - Analyze writing quality, grammar, and professional standards
   - Identify loaded language, hyperbole, and propaganda techniques
   - Assess logical flow and coherence

2. SOURCE CREDIBILITY:
   - Evaluate domain reputation and editorial standards
   - Check author credentials and expertise
   - Assess publication history and track record
   - Identify potential conflicts of interest

3. FACTUAL VERIFICATION:
   - Analyze verifiable claims and evidence quality
   - Check for citations, references, and supporting data
   - Identify unsubstantiated assertions
   - Cross-reference with known facts

4. BIAS DETECTION:
   - Identify political, ideological, or commercial bias
   - Detect one-sided reporting and missing perspectives
   - Analyze framing and selective information presentation
   - Assess balance and fairness

5. TEMPORAL ANALYSIS:
   - Check timeline consistency and plausibility
   - Verify dates, sequences, and chronological accuracy
   - Identify anachronisms or temporal inconsistencies

6. CITATION ANALYSIS:
   - Evaluate quality and relevance of sources cited
   - Check for circular references or unreliable sources
   - Assess transparency in sourcing

SCORING GUIDELINES:
- Overall Score: Weighted composite (Credibility 30%, Factuality 25%, Bias 20%, Source 25%)
- Credibility Score: Professional standards, author expertise, publication quality
- Bias Score: Higher = more biased (0 = neutral, 100 = extremely biased)
- Factuality Score: Evidence quality, verifiable claims, accuracy
- Source Reliability: Domain reputation, editorial standards, track record
- Confidence: How certain you are about your assessment

VERDICT CRITERIA:
- AUTHENTIC (70-100%): High credibility, factual accuracy, minimal bias, reliable sources
- SUSPICIOUS (40-69%): Mixed signals, some concerns, requires additional verification
- LIKELY_FAKE (0-39%): Multiple red flags, poor credibility, high bias, unreliable

METADATA ANALYSIS:
- Reading Level: Grade level assessment (e.g., "College", "High School", "Middle School")
- Emotional Tone: Overall emotional character (e.g., "Neutral", "Angry", "Fearful", "Optimistic")
- Political Lean: Detected political bias (e.g., "Neutral", "Left-leaning", "Right-leaning", "Extreme")

Provide specific, actionable insights with concrete examples from the text.`

    const { object } = await generateObject({
      model: openai("gpt-4o"),
      schema: verificationSchema,
      system: systemPrompt,
      prompt: `Perform a comprehensive verification analysis of this news article:

ARTICLE TEXT:
${articleText}

${sourceUrl ? `SOURCE URL: ${sourceUrl}` : "SOURCE URL: Not provided"}

ANALYSIS TYPE: ${analysisType}

Provide a thorough, evidence-based analysis with specific examples and detailed reasoning for all scores and assessments. Focus on concrete indicators rather than general statements.`,
    })

    // Calculate analysis time
    const analysisTime = Date.now() - startTime

    // Add computed metadata
    const wordCount = articleText.split(/\s+/).length

    const result = {
      id: analysisId,
      ...object,
      metadata: {
        ...object.metadata,
        wordCount,
        analysisTime,
      },
    }

    // Store analysis result (in a real app, you'd save to database)
    console.log(`Analysis completed: ${analysisId}, Score: ${result.overallScore}%, Time: ${analysisTime}ms`)

    return Response.json(result)
  } catch (error) {
    console.error("Error verifying news:", error)
    return Response.json({ error: "Failed to verify news article" }, { status: 500 })
  }
}
