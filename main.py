
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
import sqlite3
import hashlib
from datetime import datetime
import json
from PIL import Image
from PIL.ExifTags import TAGS
import io
import base64
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class FraudDetector:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing claims"""
        conn = sqlite3.connect('fraud_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                fraud_score REAL NOT NULL,
                metadata TEXT,
                analysis_results TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                recommendation TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_metadata(self, image_path):
        """Extract EXIF metadata from image"""
        try:
            image = Image.open(image_path)
            exifdata = image.getexif()
            metadata = {}
            
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode()
                metadata[tag] = data
            
            return metadata
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_image_hash(self, image_path):
        """Calculate perceptual hash for duplicate detection"""
        image = cv2.imread(image_path)
        image = cv2.resize(image, (8, 8))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg = image.mean()
        hash_str = ''.join('1' if pixel > avg else '0' for pixel in image.flatten())
        return hash_str
    
    def detect_tampering(self, image_path):
        """Detect potential image tampering using noise analysis"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Noise analysis - calculate variance in different regions
        h, w = gray.shape
        regions = []
        region_size = min(h, w) // 4
        
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = gray[i:i+region_size, j:j+region_size]
                variance = np.var(region)
                regions.append(variance)
        
        # If variance differs significantly between regions, potential tampering
        variance_std = np.std(regions)
        tampering_score = min(variance_std / 1000, 1.0)  # Normalize to 0-1
        
        return tampering_score
    
    def check_duplicates(self, image_hash):
        """Check for duplicate images in database"""
        conn = sqlite3.connect('fraud_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM claims WHERE file_hash = ?', (image_hash,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count > 0
    
    def analyze_image(self, image_path, filename):
        """Main fraud analysis function"""
        # Extract metadata
        metadata = self.extract_metadata(image_path)
        
        # Calculate image hash
        image_hash = self.calculate_image_hash(image_path)
        
        # Check for duplicates
        is_duplicate = self.check_duplicates(image_hash)
        
        # Detect tampering
        tampering_score = self.detect_tampering(image_path)
        
        # Calculate fraud score (0-100)
        fraud_score = 0
        
        # Scoring logic
        if is_duplicate:
            fraud_score += 40
        
        fraud_score += tampering_score * 30
        
        # Metadata inconsistencies
        if 'DateTime' not in metadata or not metadata.get('DateTime'):
            fraud_score += 15
        
        if 'GPS' not in str(metadata):
            fraud_score += 10
        
        # Random variation for demo purposes
        fraud_score += np.random.uniform(0, 15)
        fraud_score = min(fraud_score, 100)
        
        # Generate recommendation
        if fraud_score < 30:
            recommendation = "APPROVE"
        elif fraud_score < 70:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"
        
        analysis_results = {
            "is_duplicate": is_duplicate,
            "tampering_score": tampering_score,
            "metadata_issues": len([k for k in ['DateTime', 'GPS'] if k not in str(metadata)]),
            "suspicious_regions": self.generate_heatmap_data(image_path)
        }
        
        return {
            "fraud_score": round(fraud_score, 2),
            "recommendation": recommendation,
            "metadata": metadata,
            "analysis_results": analysis_results,
            "image_hash": image_hash
        }
    
    def generate_heatmap_data(self, image_path):
        """Generate heatmap data for suspicious regions"""
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Generate random suspicious regions for demo
        regions = []
        for _ in range(np.random.randint(0, 4)):
            x = np.random.randint(0, w//2)
            y = np.random.randint(0, h//2)
            regions.append({"x": x, "y": y, "width": w//4, "height": h//4})
        
        return regions
    
    def save_analysis(self, filename, analysis_result):
        """Save analysis results to database"""
        conn = sqlite3.connect('fraud_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO claims (filename, file_hash, fraud_score, metadata, analysis_results, recommendation)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            analysis_result['image_hash'],
            analysis_result['fraud_score'],
            json.dumps(analysis_result['metadata']),
            json.dumps(analysis_result['analysis_results']),
            analysis_result['recommendation']
        ))
        
        conn.commit()
        conn.close()
    
    def get_dashboard_stats(self):
        """Get statistics for dashboard"""
        conn = sqlite3.connect('fraud_detection.db')
        cursor = conn.cursor()
        
        # Total claims
        cursor.execute('SELECT COUNT(*) FROM claims')
        total_claims = cursor.fetchone()[0]
        
        # Fraud ratio
        cursor.execute('SELECT COUNT(*) FROM claims WHERE fraud_score > 70')
        high_fraud_claims = cursor.fetchone()[0]
        
        # Recent claims
        cursor.execute('SELECT * FROM claims ORDER BY timestamp DESC LIMIT 10')
        recent_claims = cursor.fetchall()
        
        # Fraud trends (last 7 days)
        cursor.execute('''
            SELECT DATE(timestamp) as date, AVG(fraud_score) as avg_score
            FROM claims
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''')
        trends = cursor.fetchall()
        
        conn.close()
        
        fraud_ratio = (high_fraud_claims / total_claims * 100) if total_claims > 0 else 0
        
        return {
            "total_claims": total_claims,
            "fraud_ratio": round(fraud_ratio, 2),
            "recent_claims": recent_claims,
            "trends": trends
        }

# Initialize fraud detector
fraud_detector = FraudDetector()

@app.route('/')
def index():
    """Main dashboard page"""
    stats = fraud_detector.get_dashboard_stats()
    return render_template('index.html', stats=stats)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the image
        result = fraud_detector.analyze_image(filepath, filename)
        
        # Save analysis to database
        fraud_detector.save_analysis(filename, result)
        
        # Convert image to base64 for display
        with open(filepath, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        
        result['image_data'] = img_base64
        result['filename'] = filename
        
        return jsonify(result)
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/generate_report/<int:claim_id>')
def generate_report(claim_id):
    """Generate PDF report for a claim"""
    conn = sqlite3.connect('fraud_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM claims WHERE id = ?', (claim_id,))
    claim = cursor.fetchone()
    
    if not claim:
        return "Claim not found", 404
    
    # Create PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Insurance Fraud Detection Report", ln=1, align='C')
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Claim ID: {claim[0]}", ln=1)
    pdf.cell(200, 10, txt=f"Filename: {claim[1]}", ln=1)
    pdf.cell(200, 10, txt=f"Fraud Score: {claim[3]}%", ln=1)
    pdf.cell(200, 10, txt=f"Recommendation: {claim[6]}", ln=1)
    pdf.cell(200, 10, txt=f"Analysis Date: {claim[5]}", ln=1)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Analysis Summary:", ln=1)
    pdf.multi_cell(0, 10, txt=f"This claim has been analyzed using AI-powered fraud detection algorithms. The fraud score of {claim[3]}% indicates the likelihood of fraudulent activity.")
    
    report_path = f"report_{claim_id}.pdf"
    pdf.output(report_path)
    
    conn.close()
    
    return send_file(report_path, as_attachment=True)

@app.route('/api/dashboard')
def dashboard_api():
    """API endpoint for dashboard data"""
    stats = fraud_detector.get_dashboard_stats()
    return jsonify(stats)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/api/export_claims')
def export_claims():
    """Export all claims to CSV"""
    try:
        conn = sqlite3.connect('fraud_detection.db')
        df = pd.read_sql_query("SELECT * FROM claims", conn)
        conn.close()
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Create response
        response = app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={"Content-disposition": "attachment; filename=fraud_claims_export.csv"}
        )
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_daily_report')
def generate_daily_report():
    """Generate daily fraud detection report"""
    try:
        conn = sqlite3.connect('fraud_detection.db')
        cursor = conn.cursor()
        
        # Get today's statistics
        cursor.execute('''
            SELECT COUNT(*) as total_claims,
                   AVG(fraud_score) as avg_fraud_score,
                   COUNT(CASE WHEN recommendation = 'REJECT' THEN 1 END) as rejected_claims,
                   COUNT(CASE WHEN recommendation = 'APPROVE' THEN 1 END) as approved_claims,
                   COUNT(CASE WHEN recommendation = 'REVIEW' THEN 1 END) as review_claims
            FROM claims 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        daily_stats = cursor.fetchone()
        
        # Get hourly breakdown
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM claims 
            WHERE DATE(timestamp) = DATE('now')
            GROUP BY hour
            ORDER BY hour
        ''')
        hourly_data = cursor.fetchall()
        
        conn.close()
        
        # Create PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        
        # Header
        pdf.cell(0, 15, "Daily Fraud Detection Report", 0, 1, 'C')
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
        pdf.ln(10)
        
        # Summary statistics
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "SUMMARY STATISTICS", 0, 1, 'L')
        pdf.ln(5)
        
        if daily_stats[0] > 0:
            pdf.cell(0, 8, f"Total Claims Processed: {daily_stats[0]}", 0, 1)
            pdf.cell(0, 8, f"Average Fraud Score: {daily_stats[1]:.2f}%", 0, 1)
            pdf.cell(0, 8, f"Approved Claims: {daily_stats[4]}", 0, 1)
            pdf.cell(0, 8, f"Claims Under Review: {daily_stats[2]}", 0, 1)
            pdf.cell(0, 8, f"Rejected Claims: {daily_stats[3]}", 0, 1)
        else:
            pdf.cell(0, 8, "No claims processed today.", 0, 1)
        
        pdf.ln(10)
        
        # Hourly breakdown
        if hourly_data:
            pdf.cell(0, 10, "HOURLY ACTIVITY", 0, 1, 'L')
            pdf.ln(5)
            for hour, count in hourly_data:
                pdf.cell(0, 8, f"{hour}:00 - {count} claims", 0, 1)
        
        # Save and return PDF
        report_filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        pdf.output(report_filename)
        
        return send_file(report_filename, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_analytics_report')
def generate_analytics_report():
    """Generate comprehensive analytics report with charts"""
    try:
        conn = sqlite3.connect('fraud_detection.db')
        
        # Get comprehensive analytics data
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute('''
            SELECT COUNT(*) as total_claims,
                   AVG(fraud_score) as avg_fraud_score,
                   MIN(fraud_score) as min_fraud_score,
                   MAX(fraud_score) as max_fraud_score,
                   COUNT(CASE WHEN fraud_score > 70 THEN 1 END) as high_risk,
                   COUNT(CASE WHEN fraud_score BETWEEN 30 AND 70 THEN 1 END) as medium_risk,
                   COUNT(CASE WHEN fraud_score < 30 THEN 1 END) as low_risk
            FROM claims
        ''')
        overall_stats = cursor.fetchone()
        
        # Weekly trends
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as claims, AVG(fraud_score) as avg_score
            FROM claims
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''')
        weekly_trends = cursor.fetchall()
        
        # Top fraudulent patterns
        cursor.execute('''
            SELECT filename, fraud_score, recommendation, timestamp
            FROM claims
            WHERE fraud_score > 70
            ORDER BY fraud_score DESC
            LIMIT 10
        ''')
        top_fraud_cases = cursor.fetchall()
        
        conn.close()
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fraud Detection Analytics Report', fontsize=16)
        
        # Risk distribution pie chart
        if overall_stats[0] > 0:
            risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_values = [overall_stats[6], overall_stats[5], overall_stats[4]]
            colors = ['#22C55E', '#EAB308', '#EF4444']
            ax1.pie(risk_values, labels=risk_labels, colors=colors, autopct='%1.1f%%')
            ax1.set_title('Risk Distribution')
            
            # Weekly trends line chart
            if weekly_trends:
                dates = [trend[0] for trend in weekly_trends]
                claims_count = [trend[1] for trend in weekly_trends]
                avg_scores = [trend[2] for trend in weekly_trends]
                
                ax2.plot(dates, claims_count, marker='o', label='Claims Count')
                ax2_twin = ax2.twinx()
                ax2_twin.plot(dates, avg_scores, marker='s', color='red', label='Avg Fraud Score')
                ax2.set_title('Weekly Trends')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Claims Count')
                ax2_twin.set_ylabel('Avg Fraud Score')
                ax2.tick_params(axis='x', rotation=45)
                
            # Fraud score distribution histogram
            cursor = conn.cursor()
            cursor.execute('SELECT fraud_score FROM claims')
            fraud_scores = [row[0] for row in cursor.fetchall()]
            
            if fraud_scores:
                ax3.hist(fraud_scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                ax3.set_title('Fraud Score Distribution')
                ax3.set_xlabel('Fraud Score')
                ax3.set_ylabel('Frequency')
                
            # Processing time simulation (since we don't track actual processing time)
            processing_times = np.random.normal(2.5, 0.5, 100)  # Simulated data
            ax4.boxplot(processing_times)
            ax4.set_title('Processing Time Analysis')
            ax4.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        chart_filename = f"analytics_chart_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create PDF report with analytics
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        
        # Header
        pdf.cell(0, 15, "Fraud Detection Analytics Report", 0, 1, 'C')
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        pdf.ln(10)
        
        # Summary statistics
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "COMPREHENSIVE ANALYTICS", 0, 1, 'L')
        pdf.ln(5)
        
        if overall_stats[0] > 0:
            pdf.cell(0, 8, f"Total Claims Analyzed: {overall_stats[0]}", 0, 1)
            pdf.cell(0, 8, f"Average Fraud Score: {overall_stats[1]:.2f}%", 0, 1)
            pdf.cell(0, 8, f"Fraud Score Range: {overall_stats[2]:.2f}% - {overall_stats[3]:.2f}%", 0, 1)
            pdf.cell(0, 8, f"High Risk Claims: {overall_stats[4]} ({(overall_stats[4]/overall_stats[0]*100):.1f}%)", 0, 1)
            pdf.cell(0, 8, f"Medium Risk Claims: {overall_stats[5]} ({(overall_stats[5]/overall_stats[0]*100):.1f}%)", 0, 1)
            pdf.cell(0, 8, f"Low Risk Claims: {overall_stats[6]} ({(overall_stats[6]/overall_stats[0]*100):.1f}%)", 0, 1)
            
            pdf.ln(10)
            
            # Add chart to PDF
            pdf.cell(0, 10, "VISUAL ANALYTICS", 0, 1, 'L')
            pdf.ln(5)
            pdf.image(chart_filename, x=10, y=None, w=190)
            
        else:
            pdf.cell(0, 8, "No claims data available for analysis.", 0, 1)
        
        # Clean up chart file
        if os.path.exists(chart_filename):
            os.remove(chart_filename)
        
        report_filename = f"analytics_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        pdf.output(report_filename)
        
        return send_file(report_filename, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_risk_assessment')
def generate_risk_assessment():
    """Generate comprehensive risk assessment report"""
    try:
        conn = sqlite3.connect('fraud_detection.db')
        cursor = conn.cursor()
        
        # Risk metrics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_claims,
                COUNT(CASE WHEN fraud_score > 90 THEN 1 END) as critical_risk,
                COUNT(CASE WHEN fraud_score BETWEEN 70 AND 90 THEN 1 END) as high_risk,
                COUNT(CASE WHEN fraud_score BETWEEN 30 AND 70 THEN 1 END) as medium_risk,
                COUNT(CASE WHEN fraud_score < 30 THEN 1 END) as low_risk,
                AVG(CASE WHEN recommendation = 'REJECT' THEN 1.0 ELSE 0.0 END) as rejection_rate,
                AVG(fraud_score) as avg_fraud_score
            FROM claims
        ''')
        risk_metrics = cursor.fetchone()
        
        # Recent high-risk claims
        cursor.execute('''
            SELECT filename, fraud_score, recommendation, timestamp
            FROM claims
            WHERE fraud_score > 70
            ORDER BY timestamp DESC
            LIMIT 15
        ''')
        high_risk_claims = cursor.fetchall()
        
        # Fraud patterns analysis
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as claims,
                AVG(fraud_score) as avg_score,
                MAX(fraud_score) as max_score
            FROM claims
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        ''')
        pattern_data = cursor.fetchall()
        
        conn.close()
        
        # Create PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        
        # Header
        pdf.cell(0, 15, "Comprehensive Risk Assessment Report", 0, 1, 'C')
        pdf.cell(0, 10, f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        pdf.ln(10)
        
        # Risk Overview
        pdf.set_font("Arial", size=14)
        pdf.cell(0, 10, "RISK OVERVIEW", 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font("Arial", size=12)
        if risk_metrics[0] > 0:
            total_claims = risk_metrics[0]
            
            # Overall risk assessment
            if risk_metrics[1] > 0:  # Critical risk claims
                pdf.set_text_color(220, 20, 60)  # Crimson
                pdf.cell(0, 8, f"⚠ CRITICAL: {risk_metrics[1]} claims with >90% fraud score", 0, 1)
                pdf.set_text_color(0, 0, 0)  # Reset to black
            
            pdf.cell(0, 8, f"Total Claims Assessed: {total_claims}", 0, 1)
            pdf.cell(0, 8, f"Critical Risk (>90%): {risk_metrics[1]} ({risk_metrics[1]/total_claims*100:.1f}%)", 0, 1)
            pdf.cell(0, 8, f"High Risk (70-90%): {risk_metrics[2]} ({risk_metrics[2]/total_claims*100:.1f}%)", 0, 1)
            pdf.cell(0, 8, f"Medium Risk (30-70%): {risk_metrics[3]} ({risk_metrics[3]/total_claims*100:.1f}%)", 0, 1)
            pdf.cell(0, 8, f"Low Risk (<30%): {risk_metrics[4]} ({risk_metrics[4]/total_claims*100:.1f}%)", 0, 1)
            pdf.cell(0, 8, f"Overall Rejection Rate: {risk_metrics[5]*100:.1f}%", 0, 1)
            pdf.cell(0, 8, f"Average Fraud Score: {risk_metrics[6]:.2f}%", 0, 1)
            
            pdf.ln(10)
            
            # Risk Recommendations
            pdf.set_font("Arial", size=14)
            pdf.cell(0, 10, "RISK RECOMMENDATIONS", 0, 1, 'L')
            pdf.ln(5)
            
            pdf.set_font("Arial", size=11)
            
            # Generate recommendations based on data
            if risk_metrics[5] > 0.3:  # High rejection rate
                pdf.cell(0, 6, "• HIGH ALERT: Elevated fraud activity detected", 0, 1)
                pdf.cell(0, 6, "  - Consider implementing additional verification steps", 0, 1)
                pdf.cell(0, 6, "  - Increase manual review for medium-risk claims", 0, 1)
            
            if risk_metrics[1] > 0:  # Critical risk cases
                pdf.cell(0, 6, "• IMMEDIATE ACTION: Critical risk claims require investigation", 0, 1)
                pdf.cell(0, 6, "  - Review critical cases for potential fraud patterns", 0, 1)
                pdf.cell(0, 6, "  - Consider legal action for confirmed fraud cases", 0, 1)
            
            if risk_metrics[6] > 50:  # High average fraud score
                pdf.cell(0, 6, "• SYSTEM ALERT: Above-average fraud scores detected", 0, 1)
                pdf.cell(0, 6, "  - Review AI model calibration", 0, 1)
                pdf.cell(0, 6, "  - Consider adjusting risk thresholds", 0, 1)
            else:
                pdf.cell(0, 6, "• NORMAL OPERATIONS: Fraud levels within expected range", 0, 1)
                pdf.cell(0, 6, "  - Continue monitoring with current procedures", 0, 1)
            
            pdf.ln(10)
            
            # High-Risk Cases Summary
            if high_risk_claims:
                pdf.set_font("Arial", size=14)
                pdf.cell(0, 10, "HIGH-RISK CASES SUMMARY", 0, 1, 'L')
                pdf.ln(5)
                
                pdf.set_font("Arial", size=10)
                for i, claim in enumerate(high_risk_claims[:10]):  # Limit to 10 for space
                    pdf.cell(0, 5, f"{i+1}. {claim[0][:30]}... - Score: {claim[1]:.1f}% ({claim[2]}) - {claim[3][:10]}", 0, 1)
        else:
            pdf.cell(0, 8, "No claims data available for risk assessment.", 0, 1)
        
        report_filename = f"risk_assessment_{datetime.now().strftime('%Y%m%d')}.pdf"
        pdf.output(report_filename)
        
        return send_file(report_filename, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
