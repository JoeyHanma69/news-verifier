#!/usr/bin/env python3
"""
Advanced Image Verification for News Articles
Run with: python scripts/image_verification.py --image_url <URL>
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
import requests
import io
import argparse
import json
from datetime import datetime
import hashlib
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ImageVerificationSystem:
    def __init__(self):
        self.results = {}
        
    def download_image(self, url):
        """Download image from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            return image, opencv_image, response.content
            
        except Exception as e:
            raise Exception(f"Failed to download image: {str(e)}")
    
    def analyze_metadata(self, image_data):
        """Analyze EXIF metadata for authenticity indicators"""
        try:
            image = Image.open(io.BytesIO(image_data))
            exifdata = image.getexif()
            
            metadata = {}
            for tag_id in exifdata:
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode('utf-8', errors='ignore')
                metadata[tag] = str(data)
            
            # Analyze authenticity indicators
            authenticity_score = self.calculate_metadata_authenticity(metadata)
            
            return {
                'metadata': metadata,
                'authenticity_score': authenticity_score,
                'has_camera_info': any(key in metadata for key in ['Make', 'Model']),
                'has_gps': 'GPSInfo' in metadata,
                'has_timestamp': 'DateTime' in metadata,
                'software_modified': 'Software' in metadata and 'Adobe' in str(metadata.get('Software', ''))
            }
            
        except Exception as e:
            return {'error': f"Metadata analysis failed: {str(e)}"}
    
    def calculate_metadata_authenticity(self, metadata):
        """Calculate authenticity score based on metadata"""
        score = 1.0
        
        # Reduce score if no camera information
        if not any(key in metadata for key in ['Make', 'Model']):
            score -= 0.3
        
        # Reduce score if software modification detected
        if 'Software' in metadata:
            software = str(metadata['Software']).lower()
            if any(editor in software for editor in ['photoshop', 'gimp', 'paint']):
                score -= 0.4
        
        # Increase score if GPS and timestamp present
        if 'GPSInfo' in metadata:
            score += 0.1
        if 'DateTime' in metadata:
            score += 0.1
        
        return max(0, min(1, score))
    
    def detect_manipulation(self, image):
        """Detect image manipulation using computer vision techniques"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Error Level Analysis (ELA) simulation
            ela_score = self.error_level_analysis(image)
            
            # Noise analysis
            noise_score = self.analyze_noise_patterns(gray)
            
            # Edge inconsistency detection
            edge_score = self.detect_edge_inconsistencies(gray)
            
            # JPEG compression artifacts
            compression_score = self.analyze_compression_artifacts(image)
            
            # Combine scores
            manipulation_probability = np.mean([ela_score, noise_score, edge_score, compression_score])
            
            return {
                'manipulation_probability': float(manipulation_probability),
                'ela_score': float(ela_score),
                'noise_score': float(noise_score),
                'edge_score': float(edge_score),
                'compression_score': float(compression_score),
                'details': {
                    'ela': 'Error Level Analysis indicates potential manipulation areas',
                    'noise': 'Noise pattern analysis for inconsistencies',
                    'edges': 'Edge detection for unnatural boundaries',
                    'compression': 'JPEG compression artifact analysis'
                }
            }
            
        except Exception as e:
            return {'error': f"Manipulation detection failed: {str(e)}"}
    
    def error_level_analysis(self, image):
        """Simulate Error Level Analysis"""
        try:
            # Convert to JPEG and back to simulate compression
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            compressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
            # Calculate difference
            diff = cv2.absdiff(image, compressed)
            
            # Calculate ELA score
            ela_score = np.mean(diff) / 255.0
            
            return min(ela_score * 2, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.5
    
    def analyze_noise_patterns(self, gray_image):
        """Analyze noise patterns for inconsistencies"""
        try:
            # Apply Gaussian blur and calculate difference
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            noise = cv2.absdiff(gray_image, blurred)
            
            # Calculate noise statistics
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            
            # Detect regions with unusual noise patterns
            threshold = noise_mean + 2 * noise_std
            unusual_noise = np.sum(noise > threshold) / noise.size
            
            return min(unusual_noise * 5, 1.0)  # Normalize
            
        except Exception:
            return 0.5
    
    def detect_edge_inconsistencies(self, gray_image):
        """Detect edge inconsistencies that might indicate manipulation"""
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Analyze edge density in different regions
            h, w = edges.shape
            regions = [
                edges[:h//2, :w//2],  # Top-left
                edges[:h//2, w//2:],  # Top-right
                edges[h//2:, :w//2],  # Bottom-left
                edges[h//2:, w//2:]   # Bottom-right
            ]
            
            densities = [np.sum(region > 0) / region.size for region in regions]
            
            # Calculate variance in edge density
            density_variance = np.var(densities)
            
            return min(density_variance * 10, 1.0)  # Normalize
            
        except Exception:
            return 0.5
    
    def analyze_compression_artifacts(self, image):
        """Analyze JPEG compression artifacts"""
        try:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            
            # Apply DCT to detect block artifacts
            h, w = y_channel.shape
            block_size = 8
            artifact_score = 0
            block_count = 0
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = y_channel[i:i+block_size, j:j+block_size].astype(np.float32)
                    
                    # Calculate block variance
                    block_var = np.var(block)
                    
                    # Low variance might indicate compression artifacts
                    if block_var < 100:  # Threshold for low variance
                        artifact_score += 1
                    
                    block_count += 1
            
            return artifact_score / block_count if block_count > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def reverse_image_search_simulation(self, image_url):
        """Simulate reverse image search (placeholder)"""
        # In production, integrate with Google Images API, TinEye, etc.
        return {
            'similar_images_found': np.random.randint(0, 10),
            'earliest_appearance': '2023-01-15',
            'search_engines_checked': ['Google', 'TinEye', 'Bing'],
            'status': 'simulation_mode'
        }
    
    def calculate_overall_authenticity(self, metadata_result, manipulation_result, reverse_search_result):
        """Calculate overall image authenticity score"""
        try:
            # Base score
            authenticity_score = 1.0
            
            # Metadata contribution (30%)
            if 'authenticity_score' in metadata_result:
                metadata_weight = 0.3
                authenticity_score -= (1 - metadata_result['authenticity_score']) * metadata_weight
            
            # Manipulation detection contribution (50%)
            if 'manipulation_probability' in manipulation_result:
                manipulation_weight = 0.5
                authenticity_score -= manipulation_result['manipulation_probability'] * manipulation_weight
            
            # Reverse search contribution (20%)
            if reverse_search_result.get('similar_images_found', 0) > 5:
                authenticity_score -= 0.2  # Many similar images might indicate stock photo
            
            return max(0, min(1, authenticity_score))
            
        except Exception:
            return 0.5
    
    def generate_report(self, image_url, metadata_result, manipulation_result, reverse_search_result, overall_score):
        """Generate comprehensive verification report"""
        report = {
            'image_url': image_url,
            'analysis_timestamp': datetime.now().isoformat(),
            'overall_authenticity_score': overall_score,
            'verdict': self.get_verdict(overall_score),
            'metadata_analysis': metadata_result,
            'manipulation_analysis': manipulation_result,
            'reverse_search': reverse_search_result,
            'recommendations': self.generate_recommendations(overall_score, metadata_result, manipulation_result)
        }
        
        return report
    
    def get_verdict(self, score):
        """Get verdict based on authenticity score"""
        if score >= 0.8:
            return "LIKELY_AUTHENTIC"
        elif score >= 0.5:
            return "UNCERTAIN"
        else:
            return "LIKELY_MANIPULATED"
    
    def generate_recommendations(self, overall_score, metadata_result, manipulation_result):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if overall_score < 0.5:
            recommendations.append("High probability of manipulation detected - verify with original source")
        
        if not metadata_result.get('has_camera_info', False):
            recommendations.append("No camera information found - could be screenshot or edited image")
        
        if metadata_result.get('software_modified', False):
            recommendations.append("Image shows signs of software editing")
        
        if manipulation_result.get('manipulation_probability', 0) > 0.7:
            recommendations.append("Multiple manipulation indicators detected")
        
        if not recommendations:
            recommendations.append("Image appears authentic based on available analysis")
        
        return recommendations
    
    def verify_image(self, image_url):
        """Main verification function"""
        try:
            print(f"Analyzing image: {image_url}")
            
            # Download image
            pil_image, cv_image, image_data = self.download_image(image_url)
            
            # Analyze metadata
            print("Analyzing metadata...")
            metadata_result = self.analyze_metadata(image_data)
            
            # Detect manipulation
            print("Detecting manipulation...")
            manipulation_result = self.detect_manipulation(cv_image)
            
            # Reverse image search
            print("Performing reverse image search...")
            reverse_search_result = self.reverse_image_search_simulation(image_url)
            
            # Calculate overall score
            overall_score = self.calculate_overall_authenticity(
                metadata_result, manipulation_result, reverse_search_result
            )
            
            # Generate report
            report = self.generate_report(
                image_url, metadata_result, manipulation_result, 
                reverse_search_result, overall_score
            )
            
            return report
            
        except Exception as e:
            return {
                'error': str(e),
                'image_url': image_url,
                'analysis_timestamp': datetime.now().isoformat()
            }

def main():
    parser = argparse.ArgumentParser(description='Verify image authenticity')
    parser.add_argument('--image_url', required=True, help='URL of the image to verify')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Initialize verification system
    verifier = ImageVerificationSystem()
    
    # Verify image
    result = verifier.verify_image(args.image_url)
    
    # Print results
    print("\n" + "="*60)
    print("IMAGE VERIFICATION RESULTS")
    print("="*60)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Overall Authenticity Score: {result['overall_authenticity_score']:.2f}")
        print(f"Verdict: {result['verdict']}")
        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"- {rec}")
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
