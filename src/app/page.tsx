import Image from "next/image" 
import { useState } from "react"
import { AlertTriangle, CheckCircle, XCircle, Loader2, Shield, Eye, TrendingUp, Link, Database } from "lucide-react" 

interface VerficationResult { 
  id: String 
  overallScore: number 
  credibilityScore: number 
  biasScore: number 
  factualityScore: number 
  sourceReliability: number 
  verdict: "AUTHENTIC" | "SUPICIOUS" | "FAKE" 
  confidence: number 
  analysis: { 
    strengths: string[] 
    concerns: string[] 
    recommendations: string[] 
    redflags: string[] 
  } 
  details: { 
    languageAnalysis: string 
    sourceAnalysis: string 
    contentAnalysis: string  
    factAnalysis: string 
    temporalAnalysis: string 
    citationalAnalysis: string 
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
  results: VerficationResult[] 
  summary: { 
    totalAnalysised: number 
    authentic: number 
    suspiscious: number
    fake: number 
    averageScore: number 
  }
} 


export default function NewsVerificationSystem() {   
  const [articleText, setArticletext] = useState(''); 
  const [sourceURL, setSourceURL] = useState("");  
  const [batchArticles, setBatchAnalysis] = useState(""); 
  const [isAnalysing, setIsAnalysing] = useState(false); 
  const [result, setResult] = useState<VerficationResult | null>(null)
  const [activeTab, setActiveTab] = useState("single") 
  const [error, setError] = useState("")

  const analyseNews = async () => { 
    if (!articleText.trim()) { 
      setError("Please enter article text to analyze") 
      return
    } 

    setIsAnalysing(true) 
    setError("") 
    setResult(null)  

    try { 
      const response = await fetch("*/api/verify-news", { 
        method: "POST",  
        headers: { 
          "Content-Type": "application/json", 
        }, 
        body: JSON.stringify({ 
          articleText, 
          sourceURL, 
          analysisType: "comprehensive",
        }),
      })  
      if (!response.ok) { 
        throw new Error("Failed to analyze article")
      } 
      const data = await response.json() 
      setResult(data)
    } catch (err) { 
      setError("Failed to analyse the article. Please try again. ") 
      console.error(err)
    } finally { 
      setIsAnalysing(false)
    }
  } 
}
