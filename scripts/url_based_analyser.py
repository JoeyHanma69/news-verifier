#!/usr/bin/env python3
"""
Batch URL Analysis Script for News Verification
Run with: python scripts/url_batch_analyzer.py --urls_file urls.txt
"""

import asyncio
import aiohttp
import argparse
import json
from datetime import datetime
from typing import List, Dict
import csv
import time

class BatchURLAnalyzer:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_single_url(self, url: str, deep_analysis: bool = True) -> Dict:
        """Analyze a single URL"""
        try:
            async with self.session.post(
                f"{self.api_base_url}/analyze/url",
                json={"url": url, "deep_analysis": deep_analysis}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        "url": url,
                        "error": f"HTTP {response.status}: {error_text}",
                        "analysis_timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def analyze_urls_from_file(self, file_path: str, deep_analysis: bool = True, delay: float = 1.0) -> List[Dict]:
        """Analyze URLs from a text file"""
        results = []
        
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            print(f"Found {len(urls)} URLs to analyze")
            
            for i, url in enumerate(urls, 1):
                print(f"Analyzing {i}/{len(urls)}: {url}")
                
                result = await self.analyze_single_url(url, deep_analysis)
                results.append(result)
                
                # Add delay between requests
                if delay > 0 and i < len(urls):
                    await asyncio.sleep(delay)
            
            return results
            
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump({
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_urls': len(results),
                    'results': results
                }, f, indent=2)
            
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def save_results_csv(self, results: List[Dict], output_file: str):
        """Save results to CSV file"""
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'URL', 'Title', 'Verdict', 'Overall Score', 'Source Score', 
                    'Content Score', 'Cross Reference Score', 'Word Count', 'Error'
                ])
                
                # Write data
                for result in results:
                    if 'error' in result:
                        writer.writerow([
                            result.get('url', ''),
                            '', '', '', '', '', '', '', result['error']
                        ])
                    else:
                        overall = result.get('overall_analysis', {})
                        content = result.get('extracted_content', {})
                        
                        writer.writerow([
                            result.get('url', ''),
                            content.get('title', ''),
                            overall.get('verdict', ''),
                            overall.get('overall_authenticity_score', ''),
                            overall.get('source_credibility_score', ''),
                            overall.get('content_quality_score', ''),
                            overall.get('cross_reference_score', ''),
                            content.get('word_count', ''),
                            ''
                        ])
            
            print(f"CSV results saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    def print_summary(self, results: List[Dict]):
        """Print analysis summary"""
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print("\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total URLs analyzed: {len(results)}")
        print(f"Successful analyses: {len(successful)}")
        print(f"Failed analyses: {len(failed)}")
        
        if successful:
            verdicts = [r['overall_analysis']['verdict'] for r in successful]
            scores = [r['overall_analysis']['overall_authenticity_score'] for r in successful]
            
            print(f"\nAverage authenticity score: {sum(scores)/len(scores):.2f}")
            print(f"Verdict distribution:")
            print(f"  - LIKELY_AUTHENTIC: {verdicts.count('LIKELY_AUTHENTIC')}")
            print(f"  - UNCERTAIN: {verdicts.count('UNCERTAIN')}")
            print(f"  - LIKELY_FAKE: {verdicts.count('LIKELY_FAKE')}")
        
        if failed:
            print(f"\nFailed URLs:")
            for result in failed[:5]:  # Show first 5 failures
                print(f"  - {result['url']}: {result['error']}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

async def main():
    parser = argparse.ArgumentParser(description='Batch analyze news URLs')
    parser.add_argument('--urls_file', required=True, help='File containing URLs to analyze (one per line)')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--csv_output', help='Output CSV file for results')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests in seconds')
    parser.add_argument('--no_deep_analysis', action='store_true', help='Skip deep analysis for faster processing')
    parser.add_argument('--api_url', default='http://localhost:8000', help='API base URL')
    
    args = parser.parse_args()
    
    deep_analysis = not args.no_deep_analysis
    
    async with BatchURLAnalyzer(args.api_url) as analyzer:
        print(f"Starting batch analysis from {args.urls_file}")
        print(f"Deep analysis: {'enabled' if deep_analysis else 'disabled'}")
        print(f"Delay between requests: {args.delay}s")
        
        start_time = time.time()
        
        results = await analyzer.analyze_urls_from_file(
            args.urls_file, 
            deep_analysis=deep_analysis,
            delay=args.delay
        )
        
        end_time = time.time()
        
        if results:
            analyzer.print_summary(results)
            
            # Save results
            if args.output:
                analyzer.save_results(results, args.output)
            
            if args.csv_output:
                analyzer.save_results_csv(results, args.csv_output)
            
            print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds")
        else:
            print("No results to save")

if __name__ == "__main__":
    asyncio.run(main())
