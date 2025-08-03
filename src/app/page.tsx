"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AlertTriangle, CheckCircle, XCircle, Loader2, Shield, Eye, TrendingUp, Link, Database } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface VerificationResult {
  id: string
  overallScore: number
  credibilityScore: number
  biasScore: number
  factualityScore: number
  sourceReliability: number
  verdict: "AUTHENTIC" | "SUSPICIOUS" | "LIKELY_FAKE"
  confidence: number
  analysis: {
    strengths: string[]
    concerns: string[]
    recommendations: string[]
    redFlags: string[]
  }
  details: {
    languageAnalysis: string
    sourceAnalysis: string
    contentAnalysis: string
    factChecking: string
    temporalAnalysis: string
    citationAnalysis: string
  }
  metadata: {
    wordCount: number
    readingLevel: string
    emotionalTone: string
    politicalLean: string
    analysisTime: number
  }
}

interface BatchResult {
  results: VerificationResult[]
  summary: {
    totalAnalyzed: number
    authentic: number
    suspicious: number
    likelyFake: number
    averageScore: number
  }
}

export default function NewsVerificationSystem() {
  const [articleText, setArticleText] = useState("")
  const [sourceUrl, setSourceUrl] = useState("")
  const [batchArticles, setBatchArticles] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<VerificationResult | null>(null)
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null)
  const [error, setError] = useState("")
  const [activeTab, setActiveTab] = useState("single")

  const analyzeNews = async () => {
    if (!articleText.trim()) {
      setError("Please enter article text to analyze")
      return
    }

    setIsAnalyzing(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("/api/verify-news", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          articleText,
          sourceUrl,
          analysisType: "comprehensive",
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to analyze article")
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError("Failed to analyze the article. Please try again.")
      console.error(err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const analyzeBatch = async () => {
    if (!batchArticles.trim()) {
      setError("Please enter articles to analyze")
      return
    }

    setIsAnalyzing(true)
    setError("")
    setBatchResult(null)

    try {
      const articles = batchArticles.split("\n---\n").filter((article) => article.trim())

      const response = await fetch("/api/verify-batch", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          articles: articles.map((article) => ({ text: article.trim() })),
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to analyze articles")
      }

      const data = await response.json()
      setBatchResult(data)
    } catch (err) {
      setError("Failed to analyze the articles. Please try again.")
      console.error(err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const analyzeFromUrl = async () => {
    if (!sourceUrl.trim()) {
      setError("Please enter a URL to analyze")
      return
    }

    setIsAnalyzing(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("/api/verify-url", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: sourceUrl,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to analyze URL")
      }

      const data = await response.json()
      setResult(data)
      setArticleText(data.extractedText || "")
    } catch (err) {
      setError("Failed to analyze the URL. Please try again.")
      console.error(err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getVerdictIcon = (verdict: string) => {
    switch (verdict) {
      case "AUTHENTIC":
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case "SUSPICIOUS":
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case "LIKELY_FAKE":
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <Shield className="h-5 w-5 text-gray-500" />
    }
  }

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case "AUTHENTIC":
        return "bg-green-100 text-green-800 border-green-200"
      case "SUSPICIOUS":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      case "LIKELY_FAKE":
        return "bg-red-100 text-red-800 border-red-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 70) return "text-green-600"
    if (score >= 40) return "text-yellow-600"
    return "text-red-600"
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Shield className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">AI News Verification System</h1>
          </div>
          <p className="text-gray-600 max-w-3xl mx-auto text-lg">
            Advanced AI-powered system to detect fake news and verify article authenticity using multiple analysis
            techniques including NLP, source credibility, and fact-checking
          </p>
        </div>

        {/* Analysis Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="single" className="flex items-center gap-2">
              <Eye className="h-4 w-4" />
              Single Article
            </TabsTrigger>
            <TabsTrigger value="url" className="flex items-center gap-2">
              <Link className="h-4 w-4" />
              From URL
            </TabsTrigger>
            <TabsTrigger value="batch" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Batch Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="single" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Eye className="h-5 w-5" />
                  Single Article Analysis
                </CardTitle>
                <CardDescription>Enter article text for comprehensive verification</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">Article Text *</label>
                  <Textarea
                    placeholder="Paste the full article text here..."
                    value={articleText}
                    onChange={(e) => setArticleText(e.target.value)}
                    className="min-h-[200px]"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">Source URL (Optional)</label>
                  <Input
                    placeholder="https://example.com/news-article"
                    value={sourceUrl}
                    onChange={(e) => setSourceUrl(e.target.value)}
                  />
                </div>
                <Button onClick={analyzeNews} disabled={isAnalyzing || !articleText.trim()} className="w-full">
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing Article...
                    </>
                  ) : (
                    <>
                      <Shield className="mr-2 h-4 w-4" />
                      Verify Article
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="url" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Link className="h-5 w-5" />
                  URL Analysis
                </CardTitle>
                <CardDescription>Enter a news article URL for automatic extraction and verification</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">News Article URL *</label>
                  <Input
                    placeholder="https://example.com/news-article"
                    value={sourceUrl}
                    onChange={(e) => setSourceUrl(e.target.value)}
                  />
                </div>
                <Button onClick={analyzeFromUrl} disabled={isAnalyzing || !sourceUrl.trim()} className="w-full">
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Extracting & Analyzing...
                    </>
                  ) : (
                    <>
                      <Link className="mr-2 h-4 w-4" />
                      Analyze URL
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="batch" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Batch Analysis
                </CardTitle>
                <CardDescription>Analyze multiple articles at once (separate articles with "---")</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">Multiple Articles *</label>
                  <Textarea
                    placeholder={`Article 1 text here...
---
Article 2 text here...
---
Article 3 text here...`}
                    value={batchArticles}
                    onChange={(e) => setBatchArticles(e.target.value)}
                    className="min-h-[300px]"
                  />
                </div>
                <Button onClick={analyzeBatch} disabled={isAnalyzing || !batchArticles.trim()} className="w-full">
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing Articles...
                    </>
                  ) : (
                    <>
                      <Database className="mr-2 h-4 w-4" />
                      Analyze Batch
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Single Article Results */}
        {result && (
          <div className="space-y-6">
            {/* Overall Verdict */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {getVerdictIcon(result.verdict)}
                  Verification Result
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <Badge className={`px-4 py-2 text-lg mb-4 ${getVerdictColor(result.verdict)}`}>
                      {result.verdict.replace("_", " ")}
                    </Badge>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Overall Score</span>
                        <span className={`font-bold ${getScoreColor(result.overallScore)}`}>
                          {result.overallScore}%
                        </span>
                      </div>
                      <Progress value={result.overallScore} className="h-3" />
                      <div className="flex justify-between text-sm text-gray-600">
                        <span>Confidence: {result.confidence}%</span>
                        <span>Analysis Time: {result.metadata.analysisTime}ms</span>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Word Count:</span>
                        <span className="ml-2 font-medium">{result.metadata.wordCount}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Reading Level:</span>
                        <span className="ml-2 font-medium">{result.metadata.readingLevel}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Emotional Tone:</span>
                        <span className="ml-2 font-medium">{result.metadata.emotionalTone}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Political Lean:</span>
                        <span className="ml-2 font-medium">{result.metadata.politicalLean}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Detailed Scores */}
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Credibility</p>
                      <p className={`text-2xl font-bold ${getScoreColor(result.credibilityScore)}`}>
                        {result.credibilityScore}%
                      </p>
                    </div>
                    <CheckCircle className="h-8 w-8 text-blue-500" />
                  </div>
                  <Progress value={result.credibilityScore} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Bias Level</p>
                      <p className={`text-2xl font-bold ${getScoreColor(100 - result.biasScore)}`}>
                        {result.biasScore}%
                      </p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-yellow-500" />
                  </div>
                  <Progress value={100 - result.biasScore} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Factuality</p>
                      <p className={`text-2xl font-bold ${getScoreColor(result.factualityScore)}`}>
                        {result.factualityScore}%
                      </p>
                    </div>
                    <Shield className="h-8 w-8 text-green-500" />
                  </div>
                  <Progress value={result.factualityScore} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Source Reliability</p>
                      <p className={`text-2xl font-bold ${getScoreColor(result.sourceReliability)}`}>
                        {result.sourceReliability}%
                      </p>
                    </div>
                    <Eye className="h-8 w-8 text-purple-500" />
                  </div>
                  <Progress value={result.sourceReliability} className="mt-2" />
                </CardContent>
              </Card>
            </div>

            {/* Analysis Grid */}
            <div className="grid md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-600">Strengths</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {result.analysis.strengths.map((strength, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{strength}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-red-600">Red Flags</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {result.analysis.redFlags.map((flag, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{flag}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-yellow-600">Concerns</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {result.analysis.concerns.map((concern, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{concern}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-blue-600">Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {result.analysis.recommendations.map((recommendation, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <Shield className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{recommendation}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>

            {/* Detailed Analysis */}
            <Card>
              <CardHeader>
                <CardTitle>Comprehensive Analysis Report</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Language Analysis</h4>
                    <p className="text-sm text-gray-600">{result.details.languageAnalysis}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Source Analysis</h4>
                    <p className="text-sm text-gray-600">{result.details.sourceAnalysis}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Content Analysis</h4>
                    <p className="text-sm text-gray-600">{result.details.contentAnalysis}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Fact Checking</h4>
                    <p className="text-sm text-gray-600">{result.details.factChecking}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Temporal Analysis</h4>
                    <p className="text-sm text-gray-600">{result.details.temporalAnalysis}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Citation Analysis</h4>
                    <p className="text-sm text-gray-600">{result.details.citationAnalysis}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Batch Results */}
        {batchResult && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Batch Analysis Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-5 gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-900">{batchResult.summary.totalAnalyzed}</div>
                    <div className="text-sm text-gray-500">Total Analyzed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{batchResult.summary.authentic}</div>
                    <div className="text-sm text-gray-500">Authentic</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-600">{batchResult.summary.suspicious}</div>
                    <div className="text-sm text-gray-500">Suspicious</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">{batchResult.summary.likelyFake}</div>
                    <div className="text-sm text-gray-500">Likely Fake</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{batchResult.summary.averageScore}%</div>
                    <div className="text-sm text-gray-500">Average Score</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid gap-4">
              {batchResult.results.map((result, index) => (
                <Card key={result.id}>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        {getVerdictIcon(result.verdict)}
                        <span className="font-medium">Article {index + 1}</span>
                        <Badge className={`${getVerdictColor(result.verdict)}`}>
                          {result.verdict.replace("_", " ")}
                        </Badge>
                      </div>
                      <div className="text-right">
                        <div className={`text-xl font-bold ${getScoreColor(result.overallScore)}`}>
                          {result.overallScore}%
                        </div>
                        <div className="text-sm text-gray-500">Overall Score</div>
                      </div>
                    </div>
                    <div className="grid md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Credibility:</span>
                        <span className={`ml-2 font-medium ${getScoreColor(result.credibilityScore)}`}>
                          {result.credibilityScore}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Bias:</span>
                        <span className={`ml-2 font-medium ${getScoreColor(100 - result.biasScore)}`}>
                          {result.biasScore}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Factuality:</span>
                        <span className={`ml-2 font-medium ${getScoreColor(result.factualityScore)}`}>
                          {result.factualityScore}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Source:</span>
                        <span className={`ml-2 font-medium ${getScoreColor(result.sourceReliability)}`}>
                          {result.sourceReliability}%
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

