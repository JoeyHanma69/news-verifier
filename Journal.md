Journal Note - AI Verification System (VerifyAI) 

02/08/2025 







About to write a idea that was presented by my classmates during my winter class. Im currently doing this as a fun project because this idea can be achieved and presented to anyone. Unfortunately, neither my team or my entrepreneurial tutor didn't want me to create it. So here I am, doing it. 



Manage to initialise the project: 

npx create-next-app@latest news-verifier --typescript --tailwind --app

cd news-verifier 





Manage to install the following dependencies: 

npm install ai @ai-sdk/openai zod lucide-react @radix-ui/react-progress 

npm install -D @types/mode 





I initialize some core ai engine and techniques that could help in this research as well.  

when implementing the verification algorithm: 

* Analyse content: language patterns, emotional manipulation, sensationalism 
* Source credibility: check domain reputation, author credentials \& publication standards 
* Fact verification: Look for citations, verifiable claims, and evidence quality 
* Bias Detection: Identify political bias, emotional language and one-sided reporting 
* Logical Consistency: Check for contradictions , timeline issues, and plausibility 



and creating a scoring system, I: 

* Overall Score (0-100%): Weighted average of all factors
* Credibility Score: Source and author reliability
* Bias Score: Level of bias detected (higher = more biased)
* Factuality Score: Factual accuracy and evidence quality
* Source Reliability: Domain and publication credibility
