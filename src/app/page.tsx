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
  const [IsAnalysing, setIsAnalysing] = useState(false); 
  const [Result, setResult] = useState("") 
  const [activeTab, setActiveTab] = useState("single") 
  
}
