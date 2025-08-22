
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
