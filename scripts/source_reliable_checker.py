#!/usr/bin/env python3
"""
Source Credibility Checker - Analyze news source domains
Run with: python scripts/source_credibility_checker.py --domain example.com
"""

import argparse
import requests
import json
from datetime import datetime
import whois
import dns.resolver
import ssl
import socket
from urllib.parse import urlparse
import re

class SourceCredibilityChecker:
    def __init__(self):
        self.reliable_sources = {
            'reuters.com': {'score': 95, 'type': 'news_agency'},
            'ap.org': {'score': 95, 'type': 'news_agency'},
            'bbc.com': {'score': 90, 'type': 'public_broadcaster'},
            'npr.org': {'score': 88, 'type': 'public_radio'},
            'pbs.org': {'score': 87, 'type': 'public_broadcaster'},
            'nytimes.com': {'score': 85, 'type': 'newspaper'},
            'washingtonpost.com': {'score': 85, 'type': 'newspaper'},
            'wsj.com': {'score': 84, 'type': 'newspaper'},
            'theguardian.com': {'score': 82, 'type': 'newspaper'},
            'cnn.com': {'score': 78, 'type': 'tv_news'}
        }
        
        self.unreliable_sources = {
            'infowars.com': {'score': 10, 'reason': 'conspiracy_theories'},
            'naturalnews.com': {'score': 15, 'reason': 'pseudoscience'},
            'breitbart.com': {'score': 25, 'reason': 'extreme_bias'},
            'rt.com': {'score': 30, 'reason': 'state_propaganda'},
            'sputniknews.com': {'score': 30, 'reason': 'state_propaganda'}
        }
    
    def analyze_domain(self, domain: str) -> dict:
        """Comprehensive domain analysis"""
        domain = domain.replace('http://', '').replace('https://', '').replace('www.', '').lower()
        
        analysis = {
            'domain': domain,
            'analysis_timestamp': datetime.now().isoformat(),
            'checks_performed': []
        }
        
        # Check against known sources
        if domain in self.reliable_sources:
            analysis.update({
                'credibility_score': self.reliable_sources[domain]['score'],
                'source_type': self.reliable_sources[domain]['type'],
                'status': 'known_reliable',
                'recommendation': 'This is a known reliable news source'
            })
            return analysis
        
        if domain in self.unreliable_sources:
            analysis.update({
                'credibility_score': self.unreliable_sources[domain]['score'],
                'reason': self.unreliable_sources[domain]['reason'],
                'status': 'known_unreliable',
                'recommendation': 'This source has known credibility issues'
            })
            return analysis
        
        # Analyze unknown domain
        score = 50  # Start with neutral score
        checks = []
        
        # WHOIS analysis
        whois_result = self.check_whois(domain)
        if whois_result:
            analysis['whois_info'] = whois_result
            checks.append('whois')
            
            if whois_result.get('domain_age_days', 0) > 365 * 5:
                score += 15
                analysis['domain_age_bonus'] = 15
            elif whois_result.get('domain_age_days', 0) < 90:
                score -= 20
                analysis['new_domain_penalty'] = -20
        
        # SSL certificate check
        ssl_result = self.check_ssl(domain)
        if ssl_result:
            analysis['ssl_info'] = ssl_result
            checks.append('ssl')
            if ssl_result.get('valid', False):
                score += 5
        
        # DNS analysis
        dns_result = self.check_dns(domain)
        if dns_result:
            analysis['dns_info'] = dns_result
            checks.append('dns')
        
        # Domain name pattern analysis
        pattern_result = self.analyze_domain_patterns(domain)
        analysis['pattern_analysis'] = pattern_result
        checks.append('pattern_analysis')
        score += pattern_result.get('score_adjustment', 0)
        
        # Website structure analysis
        structure_result = self.analyze_website_structure(domain)
        if structure_result:
            analysis['website_structure'] = structure_result
            checks.append('website_structure')
            score += structure_result.get('score_adjustment', 0)
        
        analysis['checks_performed'] = checks
        analysis['credibility_score'] = max(0, min(100, score))
        analysis['status'] = 'analyzed'
        
        # Generate recommendation
        if analysis['credibility_score'] >= 80:
            analysis['recommendation'] = 'High credibility - appears to be a reliable source'
        elif analysis['credibility_score'] >= 60:
            analysis['recommendation'] = 'Moderate credibility - verify with additional sources'
        elif analysis['credibility_score'] >= 40:
            analysis['recommendation'] = 'Low credibility - use caution and cross-reference'
        else:
            analysis['recommendation'] = 'Very low credibility - likely unreliable source'
        
        return analysis
    
    def check_whois(self, domain: str) -> dict:
        """Check WHOIS information"""
        try:
            w = whois.whois(domain)
            
            result = {
                'registrar': str(w.registrar) if w.registrar else 'Unknown',
                'creation_date': str(w.creation_date) if w.creation_date else 'Unknown',
                'expiration_date': str(w.expiration_date) if w.expiration_date else 'Unknown',
                'status': str(w.status) if w.status else 'Unknown'
            }
            
            # Calculate domain age
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation_date = w.creation_date[0]
                else:
                    creation_date = w.creation_date
                
                domain_age = (datetime.now() - creation_date).days
                result['domain_age_days'] = domain_age
                result['domain_age_years'] = round(domain_age / 365.25, 1)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def check_ssl(self, domain: str) -> dict:
        """Check SSL certificate"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    
                    return {
                        'valid': True,
                        'issuer': dict(x[0] for x in cert['issuer']),
                        'subject': dict(x[0] for x in cert['subject']),
                        'version': cert['version'],
                        'not_before': cert['notBefore'],
                        'not_after': cert['notAfter']
                    }
                    
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def check_dns(self, domain: str) -> dict:
        """Check DNS records"""
        try:
            result = {}
            
            # A record
            try:
                a_records = dns.resolver.resolve(domain, 'A')
                result['a_records'] = [str(record) for record in a_records]
            except:
                result['a_records'] = []
            
            # MX record
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                result['mx_records'] = [str(record) for record in mx_records]
            except:
                result['mx_records'] = []
            
            # NS record
            try:
                ns_records = dns.resolver.resolve(domain, 'NS')
                result['ns_records'] = [str(record) for record in ns_records]
            except:
                result['ns_records'] = []
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_domain_patterns(self, domain: str) -> dict:
        """Analyze domain name patterns for suspicious indicators"""
        score_adjustment = 0
        flags = []
        
        # Check for suspicious patterns
        if re.search(r'\d{4,}', domain):
            score_adjustment -= 10
            flags.append('contains_many_numbers')
        
        if re.search(r'[a-z]{20,}', domain):
            score_adjustment -= 10
            flags.append('very_long_words')
        
        suspicious_keywords = ['fake', 'hoax', 'conspiracy', 'truth', 'exposed', 'secret']
        for keyword in suspicious_keywords:
            if keyword in domain:
                score_adjustment -= 15
                flags.append(f'suspicious_keyword_{keyword}')
        
        # Check for fake news patterns
        if re.search(r'(news|media|press|times|post|herald)(\d+|[a-z]{1,3})$', domain):
            score_adjustment -= 10
            flags.append('fake_news_pattern')
        
        # Check TLD
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        for tld in suspicious_tlds:
            if domain.endswith(tld):
                score_adjustment -= 15
                flags.append(f'suspicious_tld_{tld}')
        
        # Positive indicators
        if any(domain.endswith(tld) for tld in ['.edu', '.gov', '.org']):
            score_adjustment += 10
            flags.append('trusted_tld')
        
        return {
            'score_adjustment': score_adjustment,
            'flags': flags,
            'analysis': f"Domain pattern analysis found {len(flags)} indicators"
        }
    
    def analyze_website_structure(self, domain: str) -> dict:
        """Analyze website structure and content"""
        try:
            response = requests.get(f'https://{domain}', timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            score_adjustment = 0
            indicators = []
            
            # Check response status
            if response.status_code == 200:
                score_adjustment += 5
                indicators.append('website_accessible')
            
            # Check for HTTPS
            if response.url.startswith('https://'):
                score_adjustment += 5
                indicators.append('https_enabled')
            
            # Basic content analysis
            content = response.text.lower()
            
            # Check for professional indicators
            professional_terms = ['about us', 'contact', 'privacy policy', 'terms of service']
            found_terms = sum(1 for term in professional_terms if term in content)
            score_adjustment += found_terms * 2
            
            if found_terms > 0:
                indicators.append(f'professional_pages_{found_terms}')
            
            # Check for suspicious content
            suspicious_terms = ['click here', 'amazing', 'shocking', 'you won\'t believe']
            suspicious_count = sum(1 for term in suspicious_terms if term in content)
            score_adjustment -= suspicious_count * 2
            
            if suspicious_count > 0:
                indicators.append(f'suspicious_content_{suspicious_count}')
            
            return {
                'score_adjustment': score_adjustment,
                'indicators': indicators,
                'status_code': response.status_code,
                'final_url': response.url
            }
            
        except Exception as e:
            return {'error': str(e), 'score_adjustment': -5}

def main():
    parser = argparse.ArgumentParser(description='Analyze news source credibility')
    parser.add_argument('--domain', required=True, help='Domain to analyze')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    checker = SourceCredibilityChecker()
    
    print(f"Analyzing domain: {args.domain}")
    print("-" * 50)
    
    result = checker.analyze_domain(args.domain)
    
    # Print results
    print(f"Domain: {result['domain']}")
    print(f"Credibility Score: {result['credibility_score']}/100")
    print(f"Status: {result['status']}")
    print(f"Recommendation: {result['recommendation']}")
    
    if args.verbose:
        print(f"\nChecks performed: {', '.join(result['checks_performed'])}")
        
        if 'whois_info' in result:
            whois = result['whois_info']
            if 'domain_age_years' in whois:
                print(f"Domain age: {whois['domain_age_years']} years")
        
        if 'pattern_analysis' in result:
            pattern = result['pattern_analysis']
            if pattern['flags']:
                print(f"Pattern flags: {', '.join(pattern['flags'])}")
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
