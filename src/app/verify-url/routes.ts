import { generateObject } from "ai"
import { openai } from "@ai-sdk/openai"
import { z } from "zod"
import type { NextRequest } from "next/server"
import { createHash } from "crypto"

const urlVerificationSchema = z.object({
  overallScore: z.number().min(0).max(100),
  credibilityScore: z.number().min(0).max(100),
  biasScore: z.number().min(0).max(100),
  factualityScore: z.number().min(0).max(100),
  sourceReliability: z.number().min(0).max(100),
  confidence: z.number().min(0).max(100),
  verdict: z.enum(["AUTHENTIC", "SUSPICIOUS", "LIKELY_FAKE"]),
  extractedText: z.string(),
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

async function extractArticleContent(url: string): Promise<string> {
  try {
    // In a real implementation, you would use a web scraping service
    // For this demo, we'll simulate content extraction
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; NewsVerifier/1.0)",
      },
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const html = await response.text()

    // Simple text extraction (in production, use a proper HTML parser)
    const textContent = html
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
      .replace(/<[^>]*>/g, " ")
      .replace(/\s+/g, " ")
      .trim()

    // Extract main content (simplified approach)
    const sentences = textContent.split(/[.!?]+/)
    const mainContent = sentences
      .filter((sentence) => sentence.length > 50)
      .slice(0, 50) // Limit to first 50 substantial sentences
      .join(". ")

    return mainContent || textContent.substring(0, 5000)
  } catch (error) {
    console.error("Content extraction error:", error)
    throw new Error("Failed to extract article content from URL")
  }
}

function analyzeDomain(url: string): { domain: string; isKnownReliable: boolean; domainAge: string } {
  try {
    const domain = new URL(url).hostname.replace("www.", "")

    // Known reliable sources (simplified list)
    const reliableSources = [
      "reuters.com",
      "ap.org",
      "bbc.com",
      "npr.org",
      "pbs.org",
      "nytimes.com",
      "washingtonpost.com",
      "wsj.com",
      "theguardian.com",
      "cnn.com",
      "abcnews.go.com",
      "cbsnews.com",
      "nbcnews.com",
    ]

    const isKnownReliable = reliableSources.some((reliable) => domain.includes(reliable) || reliable.includes(domain))

    return {
      domain,
      isKnownReliable,
      domainAge: "Unknown", // In production, you'd check domain registration date
    }
  } catch {
    return {
      domain: "Invalid URL",
      isKnownReliable: false,
      domainAge: "Unknown",
    }
  }
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const { url } = await request.json()

    if (!url) {
      return Response.json({ error: "URL is required" }, { status: 400 })
    }

    // Validate URL format
    try {
      new URL(url)
    } catch {
      return Response.json({ error: "Invalid URL format" }, { status: 400 })
    }

    // Extract article content
    let extractedText: string
    try {
      extractedText = await extractArticleContent(url)
    } catch (error) {
      return Response.json(
        {
          error: "Failed to extract content from URL. Please copy and paste the article text instead.",
        },
        { status: 400 },
      )
    }

    if (!extractedText || extractedText.length < 100) {
      return Response.json(
        {
          error: "Insufficient content extracted from URL. Please verify the URL or use manual text input.",
        },
        { status: 400 },
      )
    }

    // Analyze domain
    const domainInfo = analyzeDomain(url)

    // Generate unique ID
    const articleHash = createHash("md5").update(extractedText).digest("hex")
    const analysisId = `url_${articleHash}_${Date.now()}`

    const { object } = await generateObject({
      model: openai("gpt-4o"),
      schema: urlVerificationSchema,
      system: `You are an expert fact-checker analyzing a news article extracted from a URL. Perform comprehensive verification using advanced techniques:

ENHANCED URL ANALYSIS:
1. Source Domain Evaluation: Assess the credibility of the publication domain
2. Content Extraction Quality: Account for potential extraction issues
3. URL Structure Analysis: Evaluate URL patterns for credibility indicators
4. Cross-referencing: Consider if content matches typical patterns for the source

COMPREHENSIVE FRAMEWORK:
- Language Analysis: Professional writing, emotional manipulation, sensationalism
- Source Credibility: Domain reputation, editorial standards, author expertise
- Factual Verification: Evidence quality, citations, verifiable claims
- Bias Detection: Political/ideological bias, balanced reporting
- Temporal Analysis: Timeline consistency, chronological accuracy
- Citation Analysis: Source quality and transparency

DOMAIN CONTEXT:
- Known reliable sources get credibility boost
- Unknown domains require extra scrutiny
- URL structure and patterns matter
- Consider extraction limitations

Provide detailed analysis with specific examples from the extracted content.`,
      prompt: `Analyze this news article extracted from URL for authenticity and credibility:

SOURCE URL: ${url}
DOMAIN: ${domainInfo.domain}
KNOWN RELIABLE SOURCE: ${domainInfo.isKnownReliable ? "Yes" : "No"}

EXTRACTED ARTICLE CONTENT:
${extractedText}

Provide comprehensive verification analysis considering both the source domain credibility and the extracted content quality. Account for potential content extraction limitations in your assessment.`,
    })

    // Calculate analysis time and word count
    const analysisTime = Date.now() - startTime
    const wordCount = extractedText.split(/\s+/).length

    const result = {
      id: analysisId,
      ...object,
      extractedText,
      metadata: {
        ...object.metadata,
        wordCount,
        analysisTime,
      },
    }

    console.log(`URL analysis completed: ${url}, Score: ${result.overallScore}%, Time: ${analysisTime}ms`)

    return Response.json(result)
  } catch (error) {
    console.error("Error verifying URL:", error)
    return Response.json({ error: "Failed to verify news article from URL" }, { status: 500 })
  }
}
