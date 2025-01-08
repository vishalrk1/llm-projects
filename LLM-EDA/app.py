import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import os
import plotly.express as px
import numpy as np
from typing import List, Dict

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class AIDataAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def create_system_prompt(self, df: pd.DataFrame, selected_columns: List[str] = None) -> str:
        """Creates detailed system prompt with focus on selected columns if specified"""
        analysis_cols = selected_columns if selected_columns else df.columns
        
        # Generate detailed statistics for selected columns
        stats = {
            col: {
                "type": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "missing_values": int(df[col].isnull().sum()),
                "statistics": df[col].describe().to_dict() if df[col].dtype in ['int64', 'float64'] else None
            } for col in analysis_cols
        }
        
        return f"""You are an AI that performs thorough data analysis through self-questioning reasoning.
        
        Dataset Overview:
        - Total Records: {len(df)}
        - Analyzed Columns: {', '.join(analysis_cols)}
        - Column Statistics: {json.dumps(stats, indent=2)}
        
        Approach your analysis with:
        1. Deep exploration of patterns and relationships
        2. Critical questioning of assumptions
        3. Practical business implications
        4. Clear, actionable recommendations
        
        Express your thoughts naturally, showing your reasoning process."""

    def get_analysis(self, df: pd.DataFrame, analysis_type: str, columns: List[str] = None) -> str:
        """Generates immediate analysis based on type and selected columns"""
        prompts = {
            "data_cleaning": """
            <contemplator>
            Let me analyze this dataset for cleaning needs by considering:
            1. Missing data patterns and their implications
            2. Data quality issues and potential solutions
            3. Outlier detection and handling strategies
            4. Necessary transformations and their impact
            5. Data validation requirements
            
            Walk me through your complete analysis of these aspects.
            </contemplator>
            """,
            
            "business_insights": """
            <contemplator>
            Analyze this data for business value by exploring:
            1. Key patterns and their business implications
            2. Strategic opportunities revealed by the data
            3. Potential risks and mitigation strategies
            4. Actionable recommendations for business growth
            5. Critical metrics for ongoing monitoring
            
            Provide detailed insights with supporting evidence.
            </contemplator>
            """,
            
            "model_recommendations": """
            <contemplator>
            Recommend appropriate ML/AI models by considering:
            1. The nature of the prediction/classification task
            2. Data characteristics and their influence on model selection
            3. Potential preprocessing requirements
            4. Model evaluation strategies
            5. Implementation considerations and challenges
            
            Provide detailed model recommendations with justification.
            </contemplator>
            """
        }
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.create_system_prompt(df, columns)},
                    {"role": "user", "content": prompts[analysis_type]}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in analysis: {str(e)}"

def create_streamlit_app():
    st.set_page_config(page_title="Real-time AI Data Analyzer", layout="wide")
    st.title("AI Data Analysis Platform")
    
    analyzer = AIDataAnalyzer()

    with st.sidebar:
        st.header("About This Project")
        st.markdown("""
        ### AI-Powered Data Analysis Platform
        
        This platform uses advanced AI to provide comprehensive data analysis:
        
        🔍 **Key Features:**
        - Automated Data Cleaning Recommendations
        - In-depth Business Insights
        - ML/AI Model Suggestions
        - Interactive Visualizations
        
        ### How It Works
        1. Upload your CSV file
        2. Enter your OpenAI API key
        3. Get instant analysis across multiple dimensions
        
        ### Analysis Types
        
        **Data Cleaning:**
        - Missing value analysis
        - Outlier detection
        - Data quality assessment
        
        **Business Insights:**
        - Pattern identification
        - Strategic recommendations
        - Risk analysis
        
        **Model Recommendations:**
        - AI/ML model selection
        - Feature importance
        - Implementation guidance
        """)
    
    # File upload and API key in sidebar
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df.head())
        with col2:
            st.write("Dataset Summary")
            st.write(f"Rows: {len(df)}")
            st.write(f"Columns: {len(df.columns)}")
            st.write(f"Missing Values: {df.isnull().sum().sum()}")
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["Data Cleaning", "Business Insights", "Model Recommendations"])
        
        with tab1:
            st.subheader("Data Cleaning Analysis")
            cleaning_analysis = analyzer.get_analysis(df, "data_cleaning")
            st.markdown(cleaning_analysis)
            
            # Auto-generated visualizations for data quality
            st.subheader("Data Quality Visualizations")
            fig_missing = px.imshow(df.isnull(), 
                                  title="Missing Values Heatmap",
                                  labels=dict(color="Missing"))
            st.plotly_chart(fig_missing)
        
        with tab2:
            st.subheader("Business Insights")
            business_analysis = analyzer.get_analysis(df, "business_insights")
            st.markdown(business_analysis)
            
            # Auto-generated business metrics
            if df.select_dtypes(include=[np.number]).columns.any():
                st.subheader("Key Metrics Overview")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                fig_dist = px.box(df, y=numeric_cols, title="Distribution of Numeric Variables")
                st.plotly_chart(fig_dist)
        
        with tab3:
            st.subheader("Model Recommendations")
            
            # Column selection for modeling
            target_col = st.multiselect("Select Target Column(s)", df.columns)
            feature_cols = st.multiselect("Select Feature Columns", 
                                        [col for col in df.columns if col not in target_col])
            
            get_analysis = st.button("Get Model Recommendations", key="model_button")            
            if get_analysis:
                model_analysis = analyzer.get_analysis(df, "model_recommendations", 
                                                     columns=target_col + feature_cols)
                st.markdown(model_analysis)
                
                # Auto-generated model-related visualizations
                if feature_cols:
                    st.subheader("Feature Analysis")
                    numeric_features = df[feature_cols].select_dtypes(include=[np.number])
                    if not numeric_features.empty:
                        fig_corr = px.imshow(numeric_features.corr(),
                                           title="Feature Correlation Heatmap")
                        st.plotly_chart(fig_corr)

if __name__ == "__main__":
    create_streamlit_app()