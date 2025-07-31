import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# LangChain imports (compatible with version 0.0.232)
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Flask for web interface
from flask import Flask, render_template, jsonify, request, send_file
import threading
import webbrowser

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

class FraudDetectionAnalytics:
    def __init__(self, data_path, openai_api_key=None):
        """
        Initialize the Fraud Detection Analytics System
        
        Args:
            data_path (str): Path to the CSV data file
            openai_api_key (str): OpenAI API key for LangChain (optional, can use environment variable)
        """
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.analysis_results = {}
        self.visualizations = {}
        
        # Set up OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Load and prepare data
        self.load_data()
        self.setup_agents()
        
    def load_data(self):
        """Load and prepare the fraud detection dataset"""
        print("Loading fraud detection data...")
        self.df = pd.read_csv(self.data_path)
        
        # Basic data preprocessing
        self.df['hour'] = pd.to_datetime(self.df['step'], unit='h').dt.hour
        self.df['day'] = (self.df['step'] // 24) + 1
        
        # Create transaction amount categories
        self.df['amount_category'] = pd.cut(self.df['amount'], 
                                          bins=[0, 1000, 10000, 100000, float('inf')],
                                          labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        print(f"Fraud cases: {self.df['isFraud'].sum()}")
        print(f"Normal cases: {len(self.df) - self.df['isFraud'].sum()}")
        
    def analyze_data_distribution(self):
        """Analyze data distribution and patterns"""
        analysis = {
            'basic_stats': self.df.describe(),
            'fraud_distribution': self.df['isFraud'].value_counts(),
            'transaction_types': self.df['type'].value_counts(),
            'fraud_by_type': self.df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']),
            'correlation_matrix': self.df.select_dtypes(include=[np.number]).corr()
        }
        return analysis
    
    def build_prediction_model(self):
        """Build and train fraud detection models"""
        print("Building prediction models...")
        
        # Prepare features
        categorical_features = ['type']
        numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                             'oldbalanceDest', 'newbalanceDest', 'hour', 'day']
        
        # Encode categorical variables
        le = LabelEncoder()
        X_cat = pd.DataFrame()
        for feature in categorical_features:
            X_cat[feature] = le.fit_transform(self.df[feature])
        
        # Scale numerical features
        X_num = self.scaler.fit_transform(self.df[numerical_features])
        X_num = pd.DataFrame(X_num, columns=numerical_features)
        
        # Combine features
        X = pd.concat([X_num, X_cat], axis=1)
        y = self.df['isFraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'IsolationForest': IsolationForest(contamination=0.1, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            if name == 'IsolationForest':
                model.fit(X_train)
                y_pred = model.predict(X_test)
                y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert to binary
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            if name != 'IsolationForest':
                auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_pred)
                
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'auc_score': auc_score,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
        
        self.model = results['RandomForest']['model']  # Use RandomForest as primary model
        return results
    
    def score_transaction(self, transaction_data):
        """Score a single transaction for fraud risk"""
        # Prepare transaction features
        categorical_features = ['type']
        numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                             'oldbalanceDest', 'newbalanceDest', 'hour', 'day']
        
        # Convert transaction_data to DataFrame if it's a dict
        if isinstance(transaction_data, dict):
            transaction_data = pd.DataFrame([transaction_data])
        
        # Encode categorical variables
        le = LabelEncoder()
        X_cat = pd.DataFrame()
        for feature in categorical_features:
            X_cat[feature] = le.fit_transform(transaction_data[feature])
        
        # Scale numerical features
        X_num = self.scaler.transform(transaction_data[numerical_features])
        X_num = pd.DataFrame(X_num, columns=numerical_features)
        
        # Combine features
        X = pd.concat([X_num, X_cat], axis=1)
        
        # Predict risk score
        if self.model is not None:
            risk_score = self.model.predict_proba(X)[:, 1][0]
            return risk_score
        return None
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        visualizations = {}
        
        # 1. Fraud Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Fraud vs Normal transactions
        fraud_counts = self.df['isFraud'].value_counts()
        axes[0,0].pie(fraud_counts.values, labels=['Normal', 'Fraud'], autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Distribution of Fraud vs Normal Transactions')
        
        # Fraud by transaction type
        fraud_by_type = self.df.groupby('type')['isFraud'].sum()
        axes[0,1].bar(fraud_by_type.index, fraud_by_type.values)
        axes[0,1].set_title('Fraud Cases by Transaction Type')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Amount distribution
        axes[1,0].hist(self.df[self.df['isFraud']==0]['amount'], bins=50, alpha=0.7, label='Normal', density=True)
        axes[1,0].hist(self.df[self.df['isFraud']==1]['amount'], bins=50, alpha=0.7, label='Fraud', density=True)
        axes[1,0].set_xlabel('Transaction Amount')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Transaction Amount Distribution')
        axes[1,0].legend()
        axes[1,0].set_yscale('log')
        
        # Hourly fraud pattern
        hourly_fraud = self.df.groupby('hour')['isFraud'].mean()
        axes[1,1].plot(hourly_fraud.index, hourly_fraud.values, marker='o')
        axes[1,1].set_xlabel('Hour of Day')
        axes[1,1].set_ylabel('Fraud Rate')
        axes[1,1].set_title('Fraud Rate by Hour of Day')
        
        plt.tight_layout()
        plt.savefig('static/fraud_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('static/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Advanced Analysis with Plotly
        # Transaction flow analysis
        fig_flow = go.Figure(data=go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER", "Normal", "Fraud"],
                color = "blue"
            ),
            link = dict(
                source = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                target = [5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
                value = [
                    len(self.df[(self.df['type']=='CASH_IN') & (self.df['isFraud']==0)]),
                    len(self.df[(self.df['type']=='CASH_OUT') & (self.df['isFraud']==0)]),
                    len(self.df[(self.df['type']=='DEBIT') & (self.df['isFraud']==0)]),
                    len(self.df[(self.df['type']=='PAYMENT') & (self.df['isFraud']==0)]),
                    len(self.df[(self.df['type']=='TRANSFER') & (self.df['isFraud']==0)]),
                    len(self.df[(self.df['type']=='CASH_IN') & (self.df['isFraud']==1)]),
                    len(self.df[(self.df['type']=='CASH_OUT') & (self.df['isFraud']==1)]),
                    len(self.df[(self.df['type']=='DEBIT') & (self.df['isFraud']==1)]),
                    len(self.df[(self.df['type']=='PAYMENT') & (self.df['isFraud']==1)]),
                    len(self.df[(self.df['type']=='TRANSFER') & (self.df['isFraud']==1)])
                ]
            )
        ))
        
        fig_flow.update_layout(title_text="Transaction Flow: Type to Fraud Status", font_size=10)
        fig_flow.write_html("static/transaction_flow.html")
        
        return visualizations
    
    def setup_agents(self):
        """Setup Multi-Agent Collaboration System"""
        def analyze_fraud_patterns(query: str) -> str:
            """Data Analyst Agent: Analyze fraud patterns and statistics"""
            analysis = self.analyze_data_distribution()
            fraud_rate = analysis['fraud_distribution'][1] / len(self.df) * 100
            highest_fraud_type = analysis['fraud_by_type']['mean'].idxmax()
            result = f"""
            Fraud Analysis Results:
            - Overall fraud rate: {fraud_rate:.2f}%
            - Total transactions: {len(self.df):,}
            - Fraud cases: {analysis['fraud_distribution'][1]:,}
            - Transaction type with highest fraud rate: {highest_fraud_type}
            - Fraud rate by type: {analysis['fraud_by_type']['mean'].to_dict()}
            """
            # Store analysis for other agents
            self.analysis_results['fraud_patterns'] = result
            return result
        
        def predict_fraud_risk(query: str) -> str:
            """Risk Scoring Agent: Predict fraud risk using trained models"""
            if self.model is None:
                model_results = self.build_prediction_model()
                self.analysis_results['model_performance'] = {
                    'RandomForest': {
                        'auc_score': model_results['RandomForest']['auc_score'],
                        'classification_report': model_results['RandomForest']['classification_report']
                    }
                }
                return f"""
                Model Performance Results:
                - Random Forest AUC: {model_results['RandomForest']['auc_score']:.3f}
                - Logistic Regression AUC: {model_results['LogisticRegression']['auc_score']:.3f}
                - Models trained successfully and ready for predictions
                """
            else:
                try:
                    # Parse transaction data from query (assuming JSON-like input)
                    import json
                    transaction_data = json.loads(query)
                    risk_score = self.score_transaction(transaction_data)
                    if risk_score is not None:
                        self.analysis_results['risk_score'] = risk_score
                        return f"Transaction Risk Score: {risk_score:.3f} (0 = low risk, 1 = high risk)"
                    return "Error: Unable to score transaction. Ensure model is trained."
                except json.JSONDecodeError:
                    return "Error: Invalid transaction data format. Please provide JSON data."
        
        def generate_alerts(query: str) -> str:
            """Alert Generation Agent: Create alerts for suspicious transactions"""
            risk_score = self.analysis_results.get('risk_score', 0.5)
            fraud_patterns = self.analysis_results.get('fraud_patterns', '')
            
            # Define alert thresholds
            high_risk_threshold = 0.8
            medium_risk_threshold = 0.5
            
            alert = []
            if risk_score > high_risk_threshold:
                alert.append(f"HIGH PRIORITY ALERT: Transaction risk score ({risk_score:.3f}) exceeds high-risk threshold ({high_risk_threshold})")
            elif risk_score > medium_risk_threshold:
                alert.append(f"MEDIUM PRIORITY ALERT: Transaction risk score ({risk_score:.3f}) exceeds medium-risk threshold ({medium_risk_threshold})")
            
            # Check patterns
            if 'TRANSFER' in query.upper() or 'CASH_OUT' in query.upper():
                alert.append("ALERT: Transaction type is high-risk based on historical patterns")
            
            # Use LLM to summarize alerts
            alert_template = PromptTemplate(
                input_variables=["alerts", "patterns"],
                template="Generate a concise alert summary based on the following information:\nAlerts: {alerts}\nPatterns: {patterns}"
            )
            alert_chain = LLMChain(llm=self.llm, prompt=alert_template)
            alert_summary = alert_chain.run(alerts="\n".join(alert), patterns=fraud_patterns)
            
            self.analysis_results['alerts'] = alert_summary
            return alert_summary if alert else "No suspicious activity detected."
        
        def generate_recommendations(query: str) -> str:
            """Recommendation Agent: Suggest fraud prevention methods"""
            fraud_patterns = self.analysis_results.get('fraud_patterns', '')
            risk_score = self.analysis_results.get('risk_score', None)
            alerts = self.analysis_results.get('alerts', '')
            
            recommendation_template = PromptTemplate(
                input_variables=["patterns", "risk_score", "alerts", "query"],
                template="""
                Based on the following information, provide actionable fraud prevention recommendations:
                Fraud Patterns: {patterns}
                Risk Score: {risk_score}
                Alerts: {alerts}
                User Query: {query}
                
                Recommendations should be specific, practical, and prioritized.
                """
            )
            recommendation_chain = LLMChain(llm=self.llm, prompt=recommendation_template)
            recommendations = recommendation_chain.run(
                patterns=fraud_patterns,
                risk_score=str(risk_score) if risk_score else "Not available",
                alerts=alerts,
                query=query
            )
            
            self.analysis_results['recommendations'] = recommendations
            return recommendations
        
        # Create tools for each agent
        tools = [
            Tool(
                name="Fraud Pattern Analyzer",
                func=analyze_fraud_patterns,
                description="Analyze fraud patterns and statistics in the dataset"
            ),
            Tool(
                name="Fraud Risk Predictor",
                func=predict_fraud_risk,
                description="Build and use machine learning models to predict fraud risk. Accepts JSON transaction data for scoring."
            ),
            Tool(
                name="Alert Generator",
                func=generate_alerts,
                description="Generate alerts for suspicious transactions based on risk scores and patterns"
            ),
            Tool(
                name="Recommendation Generator",
                func=generate_recommendations,
                description="Generate actionable fraud prevention recommendations based on analysis"
            )
        ]
        
        # Initialize collaborative agent system
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def query_ai_analyst(self, question: str) -> dict:
        """Query the multi-agent system for comprehensive analysis"""
        try:
            # Run the agent executor with the question
            response = self.agent_executor.run(question)
            
            # Collect results from all agents
            results = {
                'fraud_patterns': self.analysis_results.get('fraud_patterns', ''),
                'risk_score': self.analysis_results.get('risk_score', None),
                'alerts': self.analysis_results.get('alerts', ''),
                'recommendations': self.analysis_results.get('recommendations', ''),
                'agent_response': response
            }
            return results
        except Exception as e:
            return {'error': f"Error in AI analysis: {str(e)}"}
    
    def generate_pdf_report(self, filename="fraud_analysis_report.pdf"):
        """Generate comprehensive PDF report including agent insights"""
        print("Generating PDF report...")
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            alignment=1,
            spaceAfter=30
        )
        story.append(Paragraph("Fraud Detection Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        fraud_rate = (self.df['isFraud'].sum() / len(self.df)) * 100
        summary_text = f"""
        This report presents a comprehensive analysis of financial transaction data for fraud detection.
        The dataset contains {len(self.df):,} transactions with a fraud rate of {fraud_rate:.2f}%.
        Key findings include transaction type vulnerabilities, temporal patterns, and AI-generated insights.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Dataset Overview
        story.append(Paragraph("Dataset Overview", styles['Heading2']))
        
        data_table = [
            ['Metric', 'Value'],
            ['Total Transactions', f"{len(self.df):,}"],
            ['Fraud Cases', f"{self.df['isFraud'].sum():,}"],
            ['Normal Cases', f"{len(self.df) - self.df['isFraud'].sum():,}"],
            ['Fraud Rate', f"{fraud_rate:.2f}%"],
            ['Transaction Types', str(self.df['type'].nunique())],
        ]
        
        table = Table(data_table)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        
        # AI Agent Insights
        story.append(Paragraph("Multi-Agent AI Analysis", styles['Heading2']))
        ai_results = self.query_ai_analyst("Provide comprehensive fraud analysis including patterns, risk assessment, alerts, and recommendations")
        
        # Fraud Patterns
        story.append(Paragraph("Fraud Patterns", styles['Heading3']))
        story.append(Paragraph(ai_results.get('fraud_patterns', 'No patterns available'), styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Risk Scores
        story.append(Paragraph("Risk Assessment", styles['Heading3']))
        risk_score = ai_results.get('risk_score', None)
        risk_text = f"Latest Transaction Risk Score: {risk_score:.3f}" if risk_score else "No risk score available"
        story.append(Paragraph(risk_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Alerts
        story.append(Paragraph("Alerts", styles['Heading3']))
        story.append(Paragraph(ai_results.get('alerts', 'No alerts generated'), styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading3']))
        story.append(Paragraph(ai_results.get('recommendations', 'No recommendations available'), styles['Normal']))
        
        # Build PDF
        doc.build(story)
        print(f"PDF report generated: {filename}")
        return filename

# Flask Web Application
def create_flask_app(analytics_system):
    app = Flask(__name__)
    app.secret_key = 'fraud_detection_key'
    
    # Create static and templates directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    @app.route('/')
    def dashboard():
        return render_template('dashboard.html')
    
    @app.route('/api/data_overview')
    def data_overview():
        analysis = analytics_system.analyze_data_distribution()
        return jsonify({
            'total_transactions': len(analytics_system.df),
            'fraud_cases': int(analysis['fraud_distribution'][1]),
            'fraud_rate': float(analysis['fraud_distribution'][1] / len(analytics_system.df) * 100),
            'transaction_types': analysis['transaction_types'].to_dict(),
            'fraud_by_type': analysis['fraud_by_type'].to_dict()
        })
    
    @app.route('/api/visualizations')
    def get_visualizations():
        analytics_system.create_visualizations()
        return jsonify({'status': 'success', 'message': 'Visualizations created'})
    
    @app.route('/api/ai_analysis', methods=['POST'])
    def ai_analysis():
        question = request.json.get('question', 'Analyze fraud patterns in the dataset')
        response = analytics_system.query_ai_analyst(question)
        return jsonify(response)
    
    @app.route('/api/predict_model')
    def build_model():
        results = analytics_system.build_prediction_model()
        model_performance = {}
        for name, result in results.items():
            model_performance[name] = {
                'auc_score': float(result['auc_score']),
                'classification_report': result['classification_report']
            }
        return jsonify(model_performance)
    
    @app.route('/api/generate_report')
    def generate_report():
        filename = analytics_system.generate_pdf_report()
        return jsonify({'status': 'success', 'filename': filename})
    
    @app.route('/download_report')
    def download_report():
        return send_file('fraud_analysis_report.pdf', as_attachment=True)
    
    @app.route('/api/score_transaction', methods=['POST'])
    def score_transaction():
        transaction_data = request.json
        risk_score = analytics_system.score_transaction(transaction_data)
        if risk_score is not None:
            # Trigger alert and recommendation agents
            alert_result = analytics_system.query_ai_analyst(json.dumps(transaction_data))
            return jsonify({
                'risk_score': float(risk_score),
                'alerts': alert_result.get('alerts', ''),
                'recommendations': alert_result.get('recommendations', '')
            })
        return jsonify({'error': 'Unable to score transaction. Ensure model is trained.'})

    return app

# HTML Template for Dashboard
dashboard_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric { background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .metric h3 { margin: 0; color: #667eea; }
        .metric p { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .btn { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #5a6fd8; }
        .chart-container { height: 400px; margin: 20px 0; }
        .ai-chat { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .chat-input { width: 70%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .loading { display: none; color: #667eea; }
        .response { background: white; padding: 15px; border-radius: 5px; margin-top: 10px; white-space: pre-wrap; }
        .transaction-form { margin-top: 20px; }
        .transaction-form input { margin: 5px; padding: 8px; width: 200px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Fraud Detection Analytics Dashboard</h1>
            <p>AI-Powered Financial Transaction Analysis</p>
        </div>
        
        <div class="metrics" id="metrics">
            <!-- Metrics will be loaded here -->
        </div>
        
        <div class="card">
            <h2>Data Visualizations</h2>
            <button class="btn" onclick="generateVisualizations()">Generate Visualizations</button>
            <div id="visualizations">
                <img id="fraud-analysis" src="/static/fraud_analysis.png" style="max-width: 100%; display: none;">
                <img id="correlation-heatmap" src="/static/correlation_heatmap.png" style="max-width: 100%; display: none;">
            </div>
        </div>
        
        <div class="card">
            <h2>AI Fraud Analyst</h2>
            <div class="ai-chat">
                <input type="text" class="chat-input" id="question" placeholder="Ask the AI analyst about fraud patterns...">
                <button class="btn" onclick="askAI()">Ask AI</button>
                <div class="loading" id="ai-loading">AI is analyzing...</div>
                <div class="response" id="ai-response"></div>
            </div>
        </div>
        
        <div class="card">
            <h2>Real-Time Transaction Scoring</h2>
            <div class="transaction-form">
                <input type="text" id="tx-type" placeholder="Transaction Type (e.g., TRANSFER)">
                <input type="number" id="tx-amount" placeholder="Amount">
                <input type="number" id="tx-oldbalanceOrg" placeholder="Origin Old Balance">
                <input type="number" id="tx-newbalanceOrig" placeholder="Origin New Balance">
                <input type="number" id="tx-oldbalanceDest" placeholder="Destination Old Balance">
                <input type="number" id="tx-newbalanceDest" placeholder="Destination New Balance">
                <input type="number" id="tx-hour" placeholder="Hour (0-23)">
                <input type="number" id="tx-day" placeholder="Day">
                <button class="btn" onclick="scoreTransaction()">Score Transaction</button>
            </div>
            <div class="response" id="transaction-response"></div>
        </div>
        
        <div class="card">
            <h2>Machine Learning Models</h2>
            <button class="btn" onclick="buildModels()">Build Prediction Models</button>
            <div id="model-results"></div>
        </div>
        
        <div class="card">
            <h2>Generate Report</h2>
            <button class="btn" onclick="generateReport()">Generate PDF Report</button>
            <div id="report-status"></div>
            <a id="download-link" style="display: none;" class="btn" href="/download_report">Download Report</a>
        </div>
    </div>

    <script>
        // Load initial data
        loadMetrics();
        
        async function loadMetrics() {
            try {
                const response = await fetch('/api/data_overview');
                const data = await response.json();
                
                const metricsHtml = `
                    <div class="metric">
                        <h3>Total Transactions</h3>
                        <p>${data.total_transactions.toLocaleString()}</p>
                    </div>
                    <div class="metric">
                        <h3>Fraud Cases</h3>
                        <p>${data.fraud_cases.toLocaleString()}</p>
                    </div>
                    <div class="metric">
                        <h3>Fraud Rate</h3>
                        <p>${data.fraud_rate.toFixed(2)}%</p>
                    </div>
                    <div class="metric">
                        <h3>Transaction Types</h3>
                        <p>${Object.keys(data.transaction_types).length}</p>
                    </div>
                `;
                
                document.getElementById('metrics').innerHTML = metricsHtml;
            } catch (error) {
                console.error('Error loading metrics:', error);
            }
        }
        
        async function generateVisualizations() {
            try {
                await fetch('/api/visualizations');
                document.getElementById('fraud-analysis').style.display = 'block';
                document.getElementById('correlation-heatmap').style.display = 'block';
                // Refresh images with timestamp to avoid cache
                const timestamp = new Date().getTime();
                document.getElementById('fraud-analysis').src = `/static/fraud_analysis.png?t=${timestamp}`;
                document.getElementById('correlation-heatmap').src = `/static/correlation_heatmap.png?t=${timestamp}`;
            } catch (error) {
                console.error('Error generating visualizations:', error);
            }
        }
        
        async function askAI() {
            const question = document.getElementById('question').value;
            if (!question) return;
            
            document.getElementById('ai-loading').style.display = 'block';
            document.getElementById('ai-response').innerHTML = '';
            
            try {
                const response = await fetch('/api/ai_analysis', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                let responseHtml = `<strong>Agent Response:</strong> ${data.agent_response || 'No response'}`;
                if (data.fraud_patterns) responseHtml += `<br><strong>Fraud Patterns:</strong> ${data.fraud_patterns}`;
                if (data.risk_score) responseHtml += `<br><strong>Risk Score:</strong> ${data.risk_score.toFixed(3)}`;
                if (data.alerts) responseHtml += `<br><strong>Alerts:</strong> ${data.alerts}`;
                if (data.recommendations) responseHtml += `<br><strong>Recommendations:</strong> ${data.recommendations}`;
                document.getElementById('ai-response').innerHTML = responseHtml;
            } catch (error) {
                document.getElementById('ai-response').innerHTML = 'Error: ' + error.message;
            }
            
            document.getElementById('ai-loading').style.display = 'none';
        }
        
        async function scoreTransaction() {
            const transactionData = {
                type: document.getElementById('tx-type').value,
                amount: parseFloat(document.getElementById('tx-amount').value) || 0,
                oldbalanceOrg: parseFloat(document.getElementById('tx-oldbalanceOrg').value) || 0,
                newbalanceOrig: parseFloat(document.getElementById('tx-newbalanceOrig').value) || 0,
                oldbalanceDest: parseFloat(document.getElementById('tx-oldbalanceDest').value) || 0,
                newbalanceDest: parseFloat(document.getElementById('tx-newbalanceDest').value) || 0,
                hour: parseInt(document.getElementById('tx-hour').value) || 0,
                day: parseInt(document.getElementById('tx-day').value) || 1
            };
            
            document.getElementById('transaction-response').innerHTML = 'Scoring transaction...';
            
            try {
                const response = await fetch('/api/score_transaction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(transactionData)
                });
                
                const data = await response.json();
                if (data.error) {
                    document.getElementById('transaction-response').innerHTML = `Error: ${data.error}`;
                } else {
                    document.getElementById('transaction-response').innerHTML = `
                        <strong>Risk Score:</strong> ${data.risk_score.toFixed(3)}<br>
                        <strong>Alerts:</strong> ${data.alerts || 'None'}<br>
                        <strong>Recommendations:</strong> ${data.recommendations || 'None'}
                    `;
                }
            } catch (error) {
                document.getElementById('transaction-response').innerHTML = 'Error: ' + error.message;
            }
        }
        
        async function buildModels() {
            try {
                const response = await fetch('/api/predict_model');
                const data = await response.json();
                
                let resultsHtml = '<h3>Model Performance Results:</h3>';
                for (const [modelName, metrics] of Object.entries(data)) {
                    resultsHtml += `
                        <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px;">
                            <h4>${modelName}</h4>
                            <p><strong>AUC Score:</strong> ${metrics.auc_score.toFixed(3)}</p>
                            <p><strong>Precision:</strong> ${metrics.classification_report['weighted avg']['precision'].toFixed(3)}</p>
                            <p><strong>Recall:</strong> ${metrics.classification_report['weighted avg']['recall'].toFixed(3)}</p>
                            <p><strong>F1-Score:</strong> ${metrics.classification_report['weighted avg']['f1-score'].toFixed(3)}</p>
                        </div>
                    `;
                }
                
                document.getElementById('model-results').innerHTML = resultsHtml;
            } catch (error) {
                document.getElementById('model-results').innerHTML = 'Error building models: ' + error.message;
            }
        }
        
        async function generateReport() {
            document.getElementById('report-status').innerHTML = 'Generating report...';
            
            try {
                const response = await fetch('/api/generate_report');
                const data = await response.json();
                
                if (data.status === 'success') {
                    document.getElementById('report-status').innerHTML = 'Report generated successfully!';
                    document.getElementById('download-link').style.display = 'inline-block';
                }
            } catch (error) {
                document.getElementById('report-status').innerHTML = 'Error generating report: ' + error.message;
            }
        }
    </script>
</body>
</html>
"""

# Main execution function
def main():
    """
    Main function to run the Fraud Detection Analytics System
    """
    print("Initializing Fraud Detection Analytics System...")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Verify OPENAI_API_KEY is loaded
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please ensure it is set correctly.")
    else:
        print("OPENAI_API_KEY loaded successfully")
    
    # Configuration
    DATA_PATH = os.path.join('data', 'PS_20174392719_1491204439457_log.csv')
    
    # Verify data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure the data file is in the correct location.")
        return
    
    try:
        # Initialize the analytics system
        analytics = FraudDetectionAnalytics(DATA_PATH, openai_api_key)
        
        # Create Flask app
        app = create_flask_app(analytics)
        
        # Create templates directory and save dashboard template
        os.makedirs('templates', exist_ok=True)
        with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
            f.write(dashboard_template)
        
        print("System initialized successfully!")
        print("\nStarting web dashboard...")
        print("Dashboard will be available at: http://localhost:5000")
        print("\nAvailable features:")
        print("   - Real-time fraud analytics")
        print("   - AI-powered multi-agent insights")
        print("   - Interactive visualizations")
        print("   - Machine learning predictions")
        print("   - Comprehensive PDF reports")
        print("   - Real-time transaction scoring")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        print("\nTroubleshooting tips:")
        print("   - Ensure all required packages are installed")
        print("   - Verify OPENAI_API_KEY in .env file")
        print("   - Check data file path and format")

# Advanced Analytics Features
class AdvancedFraudAnalytics:
    """
    Extended analytics with advanced features like:
    - Time series analysis
    - Network analysis
    - Anomaly detection
    - Feature importance analysis
    """
    
    def __init__(self, base_analytics):
        self.base = base_analytics
        self.df = base_analytics.df
    
    def temporal_analysis(self):
        """Analyze fraud patterns over time"""
        print("Performing temporal analysis...")
        
        # Create time-based features
        self.df['timestamp'] = pd.to_datetime(self.df['step'], unit='h')
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
        
        # Fraud rate by time periods
        hourly_fraud = self.df.groupby('hour')['isFraud'].agg(['count', 'sum', 'mean'])
        daily_fraud = self.df.groupby('day')['isFraud'].agg(['count', 'sum', 'mean'])
        
        # Create temporal visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Hourly pattern
        axes[0,0].plot(hourly_fraud.index, hourly_fraud['mean'] * 100, marker='o')
        axes[0,0].set_title('Fraud Rate by Hour of Day')
        axes[0,0].set_xlabel('Hour')
        axes[0,0].set_ylabel('Fraud Rate (%)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Daily pattern
        axes[0,1].plot(daily_fraud.index, daily_fraud['mean'] * 100, marker='s')
        axes[0,1].set_title('Fraud Rate by Day')
        axes[0,1].set_xlabel('Day')
        axes[0,1].set_ylabel('Fraud Rate (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Volume vs Fraud correlation
        volume_fraud = self.df.groupby('day').agg({
            'isFraud': ['count', 'sum'],
            'amount': 'sum'
        }).reset_index()
        volume_fraud.columns = ['day', 'total_tx', 'fraud_tx', 'total_amount']
        volume_fraud['fraud_rate'] = volume_fraud['fraud_tx'] / volume_fraud['total_tx']
        
        axes[1,0].scatter(volume_fraud['total_tx'], volume_fraud['fraud_rate'] * 100, alpha=0.6)
        axes[1,0].set_title('Transaction Volume vs Fraud Rate')
        axes[1,0].set_xlabel('Daily Transaction Count')
        axes[1,0].set_ylabel('Fraud Rate (%)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Amount vs Time heatmap
        pivot_data = self.df.groupby(['day', 'hour'])['amount'].mean().unstack()
        im = axes[1,1].imshow(pivot_data.values, cmap='viridis', aspect='auto')
        axes[1,1].set_title('Average Transaction Amount by Day/Hour')
        axes[1,1].set_xlabel('Hour')
        axes[1,1].set_ylabel('Day')
        plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('static/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'hourly_patterns': hourly_fraud.to_dict(),
            'daily_patterns': daily_fraud.to_dict(),
            'peak_fraud_hour': hourly_fraud['mean'].idxmax(),
            'peak_fraud_day': daily_fraud['mean'].idxmax()
        }
    
    def feature_importance_analysis(self):
        """Analyze feature importance for fraud detection"""
        print("Analyzing feature importance...")
        
        # Prepare features for importance analysis
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Encode categorical variables
        le = LabelEncoder()
        df_encoded = self.df.copy()
        df_encoded['type_encoded'] = le.fit_transform(df_encoded['type'])
        
        # Select features
        feature_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                          'oldbalanceDest', 'newbalanceDest', 'hour', 'day', 'type_encoded']
        
        X = df_encoded[feature_columns]
        y = df_encoded['isFraud']
        
        # Train model for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance for Fraud Detection')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('static/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return importance_df.to_dict('records')
    
    def anomaly_detection_analysis(self):
        """Perform advanced anomaly detection"""
        print("Performing anomaly detection analysis...")
        
        from sklearn.ensemble import IsolationForest
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features for anomaly detection
        feature_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                          'oldbalanceDest', 'newbalanceDest']
        
        X = self.df[feature_columns].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        
        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        # Add results to dataframe
        results_df = self.df.copy()
        results_df['anomaly_score'] = iso_forest.decision_function(X_scaled)
        results_df['is_anomaly'] = anomaly_labels == -1
        results_df['cluster'] = cluster_labels
        
        # Analyze anomaly detection performance
        anomaly_performance = {
            'total_anomalies': sum(anomaly_labels == -1),
            'fraud_in_anomalies': sum((anomaly_labels == -1) & (self.df['isFraud'] == 1)),
            'anomaly_precision': sum((anomaly_labels == -1) & (self.df['isFraud'] == 1)) / sum(anomaly_labels == -1),
            'fraud_recall': sum((anomaly_labels == -1) & (self.df['isFraud'] == 1)) / sum(self.df['isFraud'] == 1)
        }
        
        # Create anomaly visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Anomaly score distribution
        axes[0,0].hist(results_df[results_df['isFraud']==0]['anomaly_score'], 
                      bins=50, alpha=0.7, label='Normal', density=True)
        axes[0,0].hist(results_df[results_df['isFraud']==1]['anomaly_score'], 
                      bins=50, alpha=0.7, label='Fraud', density=True)
        axes[0,0].set_xlabel('Anomaly Score')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Anomaly Score Distribution')
        axes[0,0].legend()
        
        # Amount vs Balance scatter with anomalies
        fraud_mask = self.df['isFraud'] == 1
        anomaly_mask = anomaly_labels == -1
        
        axes[0,1].scatter(self.df[~fraud_mask & ~anomaly_mask]['amount'], 
                         self.df[~fraud_mask & ~anomaly_mask]['oldbalanceOrg'], 
                         alpha=0.3, label='Normal', s=1)
        axes[0,1].scatter(self.df[fraud_mask]['amount'], 
                         self.df[fraud_mask]['oldbalanceOrg'], 
                         alpha=0.7, label='Fraud', s=10, c='red')
        axes[0,1].scatter(self.df[anomaly_mask & ~fraud_mask]['amount'], 
                         self.df[anomaly_mask & ~fraud_mask]['oldbalanceOrg'], 
                         alpha=0.7, label='Anomaly', s=10, c='orange')
        axes[0,1].set_xlabel('Transaction Amount')
        axes[0,1].set_ylabel('Original Balance')
        axes[0,1].set_title('Transaction Amount vs Balance (with Anomalies)')
        axes[0,1].legend()
        axes[0,1].set_xscale('log')
        axes[0,1].set_yscale('log')
        
        # Cluster analysis
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        cluster_fraud_rates = []
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            fraud_rate = sum(self.df[cluster_mask]['isFraud']) / sum(cluster_mask)
            cluster_fraud_rates.append(fraud_rate)
        
        axes[1,0].bar(unique_clusters, cluster_fraud_rates)
        axes[1,0].set_xlabel('Cluster ID')
        axes[1,0].set_ylabel('Fraud Rate')
        axes[1,0].set_title('Fraud Rate by Cluster')
        
        # Performance metrics
        metrics_text = f"""Anomaly Detection Performance:
        
        Total Anomalies: {anomaly_performance['total_anomalies']:,}
        Fraud in Anomalies: {anomaly_performance['fraud_in_anomalies']:,}
        Precision: {anomaly_performance['anomaly_precision']:.3f}
        Recall: {anomaly_performance['fraud_recall']:.3f}
        
        Clustering Results:
        Total Clusters: {len(unique_clusters)}
        Noise Points: {sum(cluster_labels == -1):,}
        """
        
        axes[1,1].text(0.1, 0.5, metrics_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='center')
        axes[1,1].set_title('Analysis Summary')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('static/anomaly_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return anomaly_performance

# Usage example and additional features
if __name__ == "__main__":
    """
    Example usage of the Fraud Detection Analytics System
    """
    
    print("""
    Fraud Detection Analytics System
    ===================================
    
    This system provides comprehensive fraud detection capabilities:
    
    1. Data Analysis & Visualization
       • Transaction pattern analysis
       • Fraud distribution insights
       • Correlation analysis
       • Temporal pattern detection
    
    2. Multi-Agent AI System
       • Data Analyst Agent for pattern analysis
       • Risk Scoring Agent for real-time transaction scoring
       • Alert Generation Agent for suspicious activity detection
       • Recommendation Agent for fraud prevention strategies
    
    3. Machine Learning Models
       • Random Forest classifier
       • Logistic Regression
       • Isolation Forest for anomaly detection
       • Model performance evaluation
    
    4. Web Dashboard
       • Interactive Flask interface
       • Real-time visualizations
       • AI chat interface
       • Real-time transaction scoring
       • Model management
    
    5. Comprehensive Reporting
       • PDF report generation
       • Executive summaries
       • Detailed analytics
       • Actionable insights
    
    6. Advanced Analytics (Bonus Features)
       • Temporal analysis
       • Feature importance analysis
       • Anomaly detection
       • Clustering analysis
    
    Setup Instructions:
    ------------------
    1. Install required packages:
       pip install langchain==0.0.232 openai pandas numpy matplotlib seaborn scikit-learn flask plotly reportlab
    
    2. Set OpenAI API key (for AI features):
       export OPENAI_API_KEY="your-api-key-here"
    
    3. Ensure data file is in correct location:
       data/PS_20174392719_1491204439457_log.csv
    
    4. Run the system:
       python fraud_detection_system.py
    
    5. Access dashboard:
       http://localhost:5000
    
    Additional Features Ideas:
    -------------------------
    • Real-time fraud scoring API
    • Integration with external databases
    • Email alerts for high-risk transactions
    • A/B testing framework for models
    • Explainable AI features
    • Multi-language support
    • Mobile-responsive design
    • Advanced statistical tests
    • Blockchain integration for audit trails
    • Integration with business intelligence tools
    
    """)
    
    # Run the main application
    main()