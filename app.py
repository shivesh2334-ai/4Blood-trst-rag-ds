# app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import pdf2image
import io
import re
import json
import os
from datetime import datetime
import base64
from typing import Dict, List, Tuple, Optional
import hashlib

# FIXED: Updated langchain imports for newer versions
try:
    # FIXED: Updated langchain imports for newer versions
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
    except ImportError:
        # Fallback if langchain not installed
        HuggingFaceEmbeddings = None
        FAISS = None
        RecursiveCharacterTextSplitter = None
        RetrievalQA = None
        PromptTemplate = None


from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError:
    RetrievalQA = None
    PromptTemplate = None

# Configure page
st.set_page_config(
    page_title="MedLab AI Analyzer - Comprehensive Blood Investigation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'parsed_values' not in st.session_state:
    st.session_state.parsed_values = {}
if 'correction_mode' not in st.session_state:
    st.session_state.correction_mode = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_category' not in st.session_state:
    st.session_state.current_category = "all"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .category-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .parameter-box {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .parameter-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .abnormal-high {
        color: #dc2626;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .abnormal-low {
        color: #2563eb;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .normal {
        color: #059669;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .critical {
        background-color: #fef2f2;
        border-left-color: #dc2626;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }
    .diagnosis-box {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        border-left: 5px solid #9333ea;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    .next-steps {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #2563eb;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    .metabolism { border-left-color: #f59e0b; }
    .hormonal { border-left-color: #ec4899; }
    .hematology { border-left-color: #8b5cf6; }
    .immunology { border-left-color: #10b981; }
    .chemistry { border-left-color: #3b82f6; }
    
    .stButton>button {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .rag-status {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #10b981;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-size: 0.9rem;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Import reference data
from medical_reference import REFERENCE_RANGES, TEST_CATEGORIES, CRITICAL_VALUES

# Initialize RAG system with error handling
@st.cache_resource
def get_rag_system():
    try:
        from rag_components import MedLabRAG
        return MedLabRAG()
    except Exception as e:
        st.warning(f"RAG system initialization failed: {e}. Running in basic mode.")
        return None

rag_system = get_rag_system()

def extract_text_from_document(uploaded_file):
    """Extract text from various document formats"""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            images = pdf2image.convert_from_bytes(uploaded_file.read())
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
        else:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return ""

def parse_lab_values(text: str) -> Dict:
    """Advanced parsing for all blood investigation types"""
    parsed_data = {}
    
    # Comprehensive patterns for all test types
    patterns = {
        # Hematology
        'RBC': r'(?:RBC|Red Blood Cell)[\s:]*(\d+\.?\d*)\s*(?:x?10\^?12|million)?',
        'Hemoglobin': r'(?:Hemoglobin|Hb|HGB)[\s:]*(\d+\.?\d*)\s*(?:g/dL|g/L)?',
        'Hematocrit': r'(?:Hematocrit|Hct|HCT)[\s:]*(\d+\.?\d*)\s*%?',
        'MCV': r'(?:MCV)[\s:]*(\d+\.?\d*)\s*(?:fL)?',
        'MCH': r'(?:MCH)[\s:]*(\d+\.?\d*)\s*(?:pg)?',
        'MCHC': r'(?:MCHC)[\s:]*(\d+\.?\d*)\s*(?:g/dL)?',
        'RDW': r'(?:RDW)[\s:]*(\d+\.?\d*)\s*%?',
        'WBC': r'(?:WBC|White Blood Cell)[\s:]*(\d+\.?\d*)\s*(?:x?10\^?9)?',
        'Platelets': r'(?:Platelets|PLT)[\s:]*(\d+)\s*(?:x?10\^?9)?',
        'MPV': r'(?:MPV)[\s:]*(\d+\.?\d*)\s*(?:fL)?',
        'Neutrophils': r'(?:Neutrophils|Neutrophil|NEUT|ANC)[\s:]*(\d+\.?\d*)',
        'Lymphocytes': r'(?:Lymphocytes|Lymphocyte|LYMPH)[\s:]*(\d+\.?\d*)',
        'Monocytes': r'(?:Monocytes|Monocyte|MONO)[\s:]*(\d+\.?\d*)',
        'Eosinophils': r'(?:Eosinophils|Eosinophil|EO)[\s:]*(\d+\.?\d*)',
        'Basophils': r'(?:Basophils|Basophil|BASO)[\s:]*(\d+\.?\d*)',
        'Reticulocytes': r'(?:Reticulocytes|Retic)[\s:]*(\d+\.?\d*)',
        'Blasts': r'(?:Blasts|Blast)[\s:]*(\d+\.?\d*)',
        
        # Liver Function
        'ALT': r'(?:ALT|SGPT|Alanine Aminotransferase)[\s:]*(\d+\.?\d*)\s*(?:U/L)?',
        'AST': r'(?:AST|SGOT|Aspartate Aminotransferase)[\s:]*(\d+\.?\d*)\s*(?:U/L)?',
        'ALP': r'(?:ALP|Alkaline Phosphatase)[\s:]*(\d+\.?\d*)\s*(?:U/L)?',
        'GGT': r'(?:GGT|Gamma GT)[\s:]*(\d+\.?\d*)\s*(?:U/L)?',
        'Total_Bilirubin': r'(?:Total Bilirubin|T\.?\s*Bili)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Direct_Bilirubin': r'(?:Direct Bilirubin|Conjugated)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Indirect_Bilirubin': r'(?:Indirect Bilirubin|Unconjugated)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Total_Protein': r'(?:Total Protein|T\.?\s*Protein)[\s:]*(\d+\.?\d*)\s*(?:g/dL)?',
        'Albumin': r'(?:Albumin|Alb)[\s:]*(\d+\.?\d*)\s*(?:g/dL)?',
        'Globulin': r'(?:Globulin)[\s:]*(\d+\.?\d*)\s*(?:g/dL)?',
        'A_G_Ratio': r'(?:A/G Ratio|Albumin/Globulin)[\s:]*(\d+\.?\d*)',
        
        # Kidney Function
        'Creatinine': r'(?:Creatinine|Creat)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'BUN': r'(?:BUN|Blood Urea Nitrogen|Urea)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'eGFR': r'(?:eGFR|Estimated GFR)[\s:]*(\d+\.?\d*)\s*(?:mL/min)?',
        'Uric_Acid': r'(?:Uric Acid|Urate)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Sodium': r'(?:Sodium|Na)[\s:]*(\d+\.?\d*)\s*(?:mEq/L)?',
        'Potassium': r'(?:Potassium|K)[\s:]*(\d+\.?\d*)\s*(?:mEq/L)?',
        'Chloride': r'(?:Chloride|Cl)[\s:]*(\d+\.?\d*)\s*(?:mEq/L)?',
        'Bicarbonate': r'(?:Bicarbonate|CO2|HCO3)[\s:]*(\d+\.?\d*)\s*(?:mEq/L)?',
        'Calcium': r'(?:Calcium|Ca)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Phosphorus': r'(?:Phosphorus|Phosphate|P)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Magnesium': r'(?:Magnesium|Mg)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        
        # Diabetes
        'Glucose_Fasting': r'(?:Fasting Glucose|FBS|Fasting Blood Sugar)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Glucose_Random': r'(?:Random Glucose|RBS)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'HbA1c': r'(?:HbA1c|A1c|Glycated Hemoglobin)[\s:]*(\d+\.?\d*)\s*%?',
        'Insulin': r'(?:Insulin|Fasting Insulin)[\s:]*(\d+\.?\d*)\s*(?:μU/mL)?',
        'C_Peptide': r'(?:C-Peptide|C Peptide)[\s:]*(\d+\.?\d*)\s*(?:ng/mL)?',
        
        # Thyroid
        'TSH': r'(?:TSH|Thyroid Stimulating Hormone)[\s:]*(\d+\.?\d*)\s*(?:μIU/mL)?',
        'T3': r'(?:T3|Triiodothyronine|Total T3)[\s:]*(\d+\.?\d*)\s*(?:ng/dL)?',
        'T4': r'(?:T4|Thyroxine|Total T4)[\s:]*(\d+\.?\d*)\s*(?:μg/dL)?',
        'Free_T3': r'(?:Free T3|FT3)[\s:]*(\d+\.?\d*)\s*(?:pg/mL)?',
        'Free_T4': r'(?:Free T4|FT4)[\s:]*(\d+\.?\d*)\s*(?:ng/dL)?',
        'Anti_TPO': r'(?:Anti-TPO|TPO Antibodies)[\s:]*(\d+\.?\d*)\s*(?:IU/mL)?',
        'Anti_Thyroglobulin': r'(?:Anti-Thyroglobulin|Tg Antibodies)[\s:]*(\d+\.?\d*)',
        
        # Lipid Profile
        'Total_Cholesterol': r'(?:Total Cholesterol|T\.?\s*Chol)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'HDL': r'(?:HDL|HDL Cholesterol)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'LDL': r'(?:LDL|LDL Cholesterol)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Triglycerides': r'(?:Triglycerides|TG)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'VLDL': r'(?:VLDL)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'Non_HDL_Cholesterol': r'(?:Non-HDL Cholesterol)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        
        # Rheumatology/Immunology
        'RF': r'(?:RF|Rheumatoid Factor)[\s:]*(\d+\.?\d*)\s*(?:IU/mL)?',
        'Anti_CCP': r'(?:Anti-CCP|CCP Antibodies)[\s:]*(\d+\.?\d*)\s*(?:U/mL)?',
        'ANA': r'(?:ANA|Antinuclear Antibody)[\s:]*([<>\d:\s\w]+)',
        'dsDNA': r'(?:Anti-dsDNA|dsDNA)[\s:]*(\d+\.?\d*)\s*(?:IU/mL)?',
        'ESR': r'(?:ESR|Erythrocyte Sedimentation Rate)[\s:]*(\d+\.?\d*)\s*(?:mm/hr)?',
        'CRP': r'(?:CRP|C-Reactive Protein)[\s:]*(\d+\.?\d*)\s*(?:mg/L)?',
        'ASO': r'(?:ASO|Anti-Streptolysin O)[\s:]*(\d+\.?\d*)\s*(?:IU/mL)?',
        
        # Coagulation
        'PT': r'(?:PT|Prothrombin Time)[\s:]*(\d+\.?\d*)\s*(?:seconds)?',
        'INR': r'(?:INR|International Normalized Ratio)[\s:]*(\d+\.?\d*)',
        'aPTT': r'(?:aPTT|APTT|Activated Partial Thromboplastin Time)[\s:]*(\d+\.?\d*)\s*(?:seconds)?',
        'Fibrinogen': r'(?:Fibrinogen)[\s:]*(\d+\.?\d*)\s*(?:mg/dL)?',
        'D_Dimer': r'(?:D-Dimer|Dimer)[\s:]*(\d+\.?\d*)\s*(?:ng/mL)?',
        
        # Tumor Markers
        'AFP': r'(?:AFP|Alpha-Fetoprotein)[\s:]*(\d+\.?\d*)\s*(?:ng/mL)?',
        'CEA': r'(?:CEA|Carcinoembryonic Antigen)[\s:]*(\d+\.?\d*)\s*(?:ng/mL)?',
        'CA_125': r'(?:CA-125|CA 125)[\s:]*(\d+\.?\d*)\s*(?:U/mL)?',
        'CA_19_9': r'(?:CA 19-9|CA19-9)[\s:]*(\d+\.?\d*)\s*(?:U/mL)?',
        'PSA': r'(?:PSA|Prostate Specific Antigen)[\s:]*(\d+\.?\d*)\s*(?:ng/mL)?',
        'CA_15_3': r'(?:CA 15-3|CA15-3)[\s:]*(\d+\.?\d*)\s*(?:U/mL)?',
        
        # Vitamins
        'Vitamin_D': r'(?:Vitamin D|25-OH Vitamin D|25\(OH\)D)[\s:]*(\d+\.?\d*)\s*(?:ng/mL)?',
        'Vitamin_B12': r'(?:Vitamin B12|B12|Cobalamin)[\s:]*(\d+\.?\d*)\s*(?:pg/mL)?',
        'Folate': r'(?:Folate|Folic Acid)[\s:]*(\d+\.?\d*)\s*(?:ng/mL)?',
        'Iron': r'(?:Iron|Serum Iron)[\s:]*(\d+\.?\d*)\s*(?:μg/dL)?',
        'Ferritin': r'(?:Ferritin)[\s:]*(\d+\.?\d*)\s*(?:ng/mL)?',
        'TIBC': r'(?:TIBC|Total Iron Binding Capacity)[\s:]*(\d+\.?\d*)\s*(?:μg/dL)?',
        'Transferrin_Saturation': r'(?:Transferrin Saturation|TSAT)[\s:]*(\d+\.?\d*)\s*%?',
    }
    
    for param, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                val = matches[0]
                if isinstance(val, tuple):
                    val = val[0]
                val = str(val).replace('<', '').replace('>', '').strip()
                parsed_data[param] = float(val) if val.replace('.','').isdigit() else val
            except:
                parsed_data[param] = matches[0]
    
    return parsed_data

def categorize_tests(tests: Dict) -> Dict[str, Dict]:
    """Categorize tests by medical system"""
    categorized = {
        'Hematology': {},
        'Liver_Function': {},
        'Kidney_Function': {},
        'Metabolic': {},
        'Endocrine': {},
        'Lipid_Profile': {},
        'Immunology_Rheumatology': {},
        'Coagulation': {},
        'Tumor_Markers': {},
        'Vitamins_Minerals': {},
        'Other': {}
    }
    
    category_map = {
        'Hematology': ['RBC', 'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC', 'RDW', 
                      'WBC', 'Platelets', 'MPV', 'Neutrophils', 'Lymphocytes', 'Monocytes',
                      'Eosinophils', 'Basophils', 'Reticulocytes', 'Blasts'],
        'Liver_Function': ['ALT', 'AST', 'ALP', 'GGT', 'Total_Bilirubin', 'Direct_Bilirubin',
                          'Indirect_Bilirubin', 'Total_Protein', 'Albumin', 'Globulin', 'A_G_Ratio'],
        'Kidney_Function': ['Creatinine', 'BUN', 'eGFR', 'Uric_Acid', 'Sodium', 'Potassium',
                           'Chloride', 'Bicarbonate', 'Calcium', 'Phosphorus', 'Magnesium'],
        'Metabolic': ['Glucose_Fasting', 'Glucose_Random', 'HbA1c', 'Insulin', 'C_Peptide'],
        'Endocrine': ['TSH', 'T3', 'T4', 'Free_T3', 'Free_T4', 'Anti_TPO', 'Anti_Thyroglobulin'],
        'Lipid_Profile': ['Total_Cholesterol', 'HDL', 'LDL', 'Triglycerides', 'VLDL', 'Non_HDL_Cholesterol'],
        'Immunology_Rheumatology': ['RF', 'Anti_CCP', 'ANA', 'dsDNA', 'ESR', 'CRP', 'ASO'],
        'Coagulation': ['PT', 'INR', 'aPTT', 'Fibrinogen', 'D_Dimer'],
        'Tumor_Markers': ['AFP', 'CEA', 'CA_125', 'CA_19_9', 'PSA', 'CA_15_3'],
        'Vitamins_Minerals': ['Vitamin_D', 'Vitamin_B12', 'Folate', 'Iron', 'Ferritin', 'TIBC', 'Transferrin_Saturation']
    }
    
    for test, value in tests.items():
        found = False
        for category, test_list in category_map.items():
            if test in test_list:
                categorized[category][test] = value
                found = True
                break
        if not found:
            categorized['Other'][test] = value
    
    return {k: v for k, v in categorized.items() if v}

def check_critical_values(tests: Dict) -> List[Dict]:
    """Check for life-threatening values"""
    criticals = []
    
    for test, value in tests.items():
        if test in CRITICAL_VALUES and isinstance(value, (int, float)):
            low, high = CRITICAL_VALUES[test]
            if value < low or value > high:
                criticals.append({
                    'test': test,
                    'value': value,
                    'range': f"{low}-{high}",
                    'direction': 'low' if value < low else 'high'
                })
    
    return criticals

def get_status_class(test: str, value: float, gender: str = 'male') -> Tuple[str, str, str]:
    """Determine status and styling for a test value"""
    if test not in REFERENCE_RANGES:
        return "normal", "✓", "Unknown reference"
    
    ref = REFERENCE_RANGES[test]
    unit = ref.get('unit', '')
    
    # Handle gender-specific ranges
    if 'male' in ref and 'female' in ref:
        low, high = ref[gender]
    else:
        low, high = ref['range']
    
    if value < low:
        return "abnormal-low", "↓", f"Low (Ref: {low}-{high} {unit})"
    elif value > high:
        return "abnormal-high", "↑", f"High (Ref: {low}-{high} {unit})"
    else:
        return "normal", "✓", f"Normal (Ref: {low}-{high} {unit})"

def display_parameter_card(test: str, value, category: str, gender: str = 'male', editable: bool = False):
    """Display a parameter card with optional editing"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**{test.replace('_', ' ')}**")
        if isinstance(value, (int, float)):
            status_class, icon, ref_text = get_status_class(test, value, gender)
            
            # Check if critical
            is_critical = test in CRITICAL_VALUES and isinstance(value, (int, float))
            if is_critical:
                crit_low, crit_high = CRITICAL_VALUES[test]
                is_critical = value < crit_low or value > crit_high
            
            critical_class = "critical" if is_critical else ""
            
            st.markdown(f"""
            <div class="parameter-box {critical_class}">
                <span class="{status_class}">{value} {REFERENCE_RANGES.get(test, {}).get('unit', '')} {icon}</span><br>
                <small>{ref_text}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='parameter-box'><strong>{value}</strong></div>", unsafe_allow_html=True)
    
    with col2:
        if editable and isinstance(value, (int, float)):
            new_value = st.number_input(f"Correct {test}", value=float(value), key=f"edit_{test}", label_visibility="collapsed")
            if new_value != value:
                st.session_state.parsed_values[test] = new_value
    
    with col3:
        if editable:
            if st.button("🗑️", key=f"del_{test}"):
                del st.session_state.parsed_values[test]
                st.rerun()

def analyze_hematology_patterns(tests: Dict) -> List[str]:
    """Recognize hematology patterns"""
    patterns = []
    
    # Anemia classification
    if 'Hemoglobin' in tests and 'MCV' in tests:
        hgb = tests['Hemoglobin']
        mcv = tests['MCV']
        
        if hgb < 12:
            if mcv < 80:
                patterns.append("🔴 Microcytic anemia - consider iron deficiency, thalassemia, or anemia of chronic disease")
            elif mcv > 100:
                patterns.append("🔴 Macrocytic anemia - consider B12/folate deficiency, liver disease, MDS, or hemolysis")
            else:
                patterns.append("🟡 Normocytic anemia - consider acute blood loss, hemolysis, or early iron deficiency")
    
    # RDW interpretation
    if 'RDW' in tests and tests['RDW'] > 14.5:
        patterns.append("📊 Elevated RDW suggests anisocytosis - seen in iron deficiency, mixed deficiencies, or post-transfusion")
    
    # Thrombocytopenia patterns
    if 'Platelets' in tests and tests['Platelets'] < 150:
        if 'MPV' in tests:
            if tests['MPV'] > 11.5:
                patterns.append("🔴 Thrombocytopenia with high MPV suggests peripheral destruction (ITP, TTP)")
            else:
                patterns.append("🔴 Thrombocytopenia with normal/low MPV suggests bone marrow failure or sequestration")
        else:
            patterns.append("🟡 Thrombocytopenia - verify with peripheral smear for pseudothrombocytopenia")
    
    # Leukocytosis patterns
    if 'WBC' in tests:
        wbc = tests['WBC']
        if wbc > 11:
            if 'Neutrophils' in tests and tests['Neutrophils'] > 70:
                patterns.append("🟡 Neutrophilic leukocytosis suggests bacterial infection, inflammation, or stress")
            elif 'Lymphocytes' in tests and tests['Lymphocytes'] > 40:
                patterns.append("🟡 Lymphocytosis suggests viral infection, lymphoid malignancy, or pertussis")
        elif wbc < 4:
            patterns.append("🔴 Leukopenia - increased infection risk, consider viral infection or bone marrow suppression")
    
    # Blasts detection
    if 'Blasts' in tests and tests['Blasts'] > 0:
        patterns.append(f"🚨 CRITICAL: {tests['Blasts']}% blasts detected - possible acute leukemia requiring immediate hematology referral")
    
    return patterns

def analyze_liver_patterns(tests: Dict) -> List[str]:
    """Recognize liver disease patterns"""
    patterns = []
    
    # Hepatocellular vs cholestatic
    if 'ALT' in tests and 'ALP' in tests:
        alt = tests['ALT']
        alp = tests['ALP']
        
        if alt > 40 and alp < 120:
            patterns.append("🔴 Hepatocellular pattern - suggests viral hepatitis, drug-induced injury, or ischemic hepatitis")
        elif alp > 120 and alt < 40:
            patterns.append("🔴 Cholestatic pattern - suggests biliary obstruction, primary biliary cholangitis, or drug-induced cholestasis")
        elif alt > 40 and alp > 120:
            patterns.append("🟡 Mixed hepatocellular-cholestatic pattern - suggests alcoholic hepatitis or acute viral hepatitis")
    
    # Bilirubin fractionation
    if 'Total_Bilirubin' in tests and 'Direct_Bilirubin' in tests:
        total = tests['Total_Bilirubin']
        direct = tests['Direct_Bilirubin']
        
        if total > 1.2:
            if direct / total > 0.5:
                patterns.append("🔴 Conjugated hyperbilirubinemia - suggests hepatocellular disease or biliary obstruction")
            else:
                patterns.append("🟡 Unconjugated hyperbilirubinemia - suggests hemolysis, Gilbert syndrome, or ineffective erythropoiesis")
    
    # Synthetic function
    if 'Albumin' in tests and tests['Albumin'] < 3.5:
        patterns.append("📉 Hypoalbuminemia suggests decreased synthetic function - chronic liver disease, malnutrition, or nephrotic syndrome")
    
    if 'INR' in tests and tests['INR'] > 1.2:
        patterns.append("🔴 Elevated INR suggests impaired coagulation factor synthesis - severe liver disease or vitamin K deficiency")
    
    return patterns

def analyze_kidney_patterns(tests: Dict) -> List[str]:
    """Recognize kidney disease patterns"""
    patterns = []
    
    # AKI vs CKD indicators
    if 'Creatinine' in tests:
        creat = tests['Creatinine']
        if creat > 1.2:
            patterns.append(f"🔴 Elevated creatinine ({creat}) suggests reduced GFR")
            
            if 'BUN' in tests:
                bun = tests['BUN']
                ratio = bun / creat
                if ratio > 20:
                    patterns.append("📊 BUN:Creatinine ratio >20 suggests prerenal azotemia (dehydration, CHF, GI bleeding)")
                elif ratio < 10:
                    patterns.append("📊 BUN:Creatinine ratio <10 suggests intrinsic renal disease or liver disease")
    
    if 'eGFR' in tests:
        egfr = tests['eGFR']
        if egfr < 60:
            stage = "G3a-G5" if egfr < 60 else "G3b" if egfr < 45 else "G4" if egfr < 30 else "G5"
            patterns.append(f"🔴 eGFR {egfr} indicates CKD {stage} - evaluate for complications")
    
    # Electrolyte disturbances
    if 'Potassium' in tests:
        k = tests['Potassium']
        if k > 5.0:
            patterns.append(f"🚨 Hyperkalemia ({k}) - risk of cardiac arrhythmia, requires urgent management")
        elif k < 3.5:
            patterns.append(f"🟡 Hypokalemia ({k}) - consider diuretic use, GI losses, or renal wasting")
    
    return patterns

def analyze_metabolic_patterns(tests: Dict) -> List[str]:
    """Analyze diabetes and metabolic patterns"""
    patterns = []
    
    if 'HbA1c' in tests:
        a1c = tests['HbA1c']
        if a1c >= 6.5:
            patterns.append(f"🔴 HbA1c {a1c}% meets criteria for diabetes mellitus")
        elif a1c >= 5.7:
            patterns.append(f"🟡 HbA1c {a1c}% indicates prediabetes - lifestyle intervention recommended")
    
    if 'Glucose_Fasting' in tests:
        glucose = tests['Glucose_Fasting']
        if glucose >= 126:
            patterns.append(f"🔴 Fasting glucose {glucose} mg/dL meets diabetes criteria")
        elif glucose >= 100:
            patterns.append(f"🟡 Impaired fasting glucose ({glucose}) - prediabetes")
    
    return patterns

def analyze_thyroid_patterns(tests: Dict) -> List[str]:
    """Analyze thyroid function patterns"""
    patterns = []
    
    if 'TSH' in tests:
        tsh = tests['TSH']
        
        if tsh > 4.5:
            if 'Free_T4' in tests:
                if tests['Free_T4'] < 0.8:
                    patterns.append("🔴 Primary hypothyroidism - elevated TSH with low FT4")
                else:
                    patterns.append("🟡 Subclinical hypothyroidism - elevated TSH with normal FT4")
            
            if 'Anti_TPO' in tests and tests['Anti_TPO'] > 35:
                patterns.append("📊 Positive Anti-TPO suggests autoimmune (Hashimoto's) thyroiditis")
                
        elif tsh < 0.4:
            if 'Free_T4' in tests:
                if tests['Free_T4'] > 1.8:
                    patterns.append("🔴 Primary hyperthyroidism - suppressed TSH with elevated FT4")
                else:
                    patterns.append("🟡 Subclinical hyperthyroidism - suppressed TSH with normal FT4")
    
    return patterns

def analyze_lipid_patterns(tests: Dict) -> List[str]:
    """Analyze lipid abnormalities"""
    patterns = []
    
    if 'LDL' in tests and tests['LDL'] > 100:
        patterns.append(f"🟡 Elevated LDL ({tests['LDL']}) - increased cardiovascular risk")
    
    if 'HDL' in tests and tests['HDL'] < 40:
        patterns.append("🟡 Low HDL - cardiovascular risk factor")
    
    if 'Triglycerides' in tests and tests['Triglycerides'] > 150:
        if tests['Triglycerides'] > 500:
            patterns.append(f"🔴 Severe hypertriglyceridemia ({tests['Triglycerides']}) - pancreatitis risk")
        else:
            patterns.append("🟡 Elevated triglycerides - metabolic syndrome component")
    
    return patterns

def analyze_rheumatology_patterns(tests: Dict) -> List[str]:
    """Analyze autoimmune and rheumatology patterns"""
    patterns = []
    
    # Rheumatoid Arthritis
    if 'RF' in tests and tests['RF'] > 20:
        patterns.append("📊 Positive RF supports rheumatoid arthritis diagnosis")
    if 'Anti_CCP' in tests and tests['Anti_CCP'] > 20:
        patterns.append("📊 Anti-CCP positive - highly specific for rheumatoid arthritis")
    
    # Lupus
    if 'ANA' in tests:
        patterns.append("📊 Positive ANA - if clinically suspected, check specific autoantibodies (dsDNA, Sm, RNP)")
    if 'dsDNA' in tests and isinstance(tests['dsDNA'], (int, float)) and tests['dsDNA'] > 100:
        patterns.append("🔴 Elevated anti-dsDNA - specific for systemic lupus erythematosus")
    
    # Inflammation
    if 'ESR' in tests and tests['ESR'] > 20:
        patterns.append(f"📊 Elevated ESR ({tests['ESR']}) indicates active inflammation")
    if 'CRP' in tests and tests['CRP'] > 10:
        patterns.append(f"📊 Elevated CRP ({tests['CRP']}) suggests acute inflammation or infection")
    
    return patterns

def generate_differential_diagnosis(categorized_tests: Dict, gender: str, age: int) -> List[Dict]:
    """Generate prioritized differential diagnoses"""
    diagnoses = []
    
    # Hematology diagnoses
    if 'Hematology' in categorized_tests:
        heme = categorized_tests['Hematology']
        
        if 'Blasts' in heme and isinstance(heme['Blasts'], (int, float)) and heme['Blasts'] > 5:
            diagnoses.append({
                'condition': 'Acute Leukemia',
                'probability': 'High',
                'urgency': 'Critical',
                'supporting_evidence': [f"{heme['Blasts']}% blasts in peripheral blood"],
                'next_step': 'Urgent hematology referral, bone marrow biopsy, flow cytometry'
            })
        
        if 'Hemoglobin' in heme and isinstance(heme['Hemoglobin'], (int, float)) and heme['Hemoglobin'] < 7:
            diagnoses.append({
                'condition': 'Severe Anemia',
                'probability': 'Confirmed',
                'urgency': 'High',
                'supporting_evidence': [f"Hemoglobin {heme['Hemoglobin']} g/dL"],
                'next_step': 'Transfusion consideration, iron studies, B12/folate, reticulocyte count'
            })
    
    # Metabolic diagnoses
    if 'Metabolic' in categorized_tests:
        metab = categorized_tests['Metabolic']
        
        if 'HbA1c' in metab and isinstance(metab['HbA1c'], (int, float)) and metab['HbA1c'] >= 6.5:
            diagnoses.append({
                'condition': 'Diabetes Mellitus',
                'probability': 'High',
                'urgency': 'Moderate',
                'supporting_evidence': [f"HbA1c {metab['HbA1c']}%"],
                'next_step': 'Confirm with repeat testing, ophthalmology referral, urine microalbumin, lipid panel'
            })
    
    # Kidney diagnoses
    if 'Kidney_Function' in categorized_tests:
        renal = categorized_tests['Kidney_Function']
        
        if 'eGFR' in renal and isinstance(renal['eGFR'], (int, float)) and renal['eGFR'] < 30:
            diagnoses.append({
                'condition': 'Stage 4-5 Chronic Kidney Disease',
                'probability': 'High',
                'urgency': 'High',
                'supporting_evidence': [f"eGFR {renal['eGFR']} mL/min"],
                'next_step': 'Nephrology referral, renal ultrasound, anemia workup, bone metabolism assessment'
            })
    
    # Liver diagnoses
    if 'Liver_Function' in categorized_tests:
        liver = categorized_tests['Liver_Function']
        
        if 'Total_Bilirubin' in liver and isinstance(liver['Total_Bilirubin'], (int, float)) and liver['Total_Bilirubin'] > 3:
            diagnoses.append({
                'condition': 'Jaundice/Hepatic Dysfunction',
                'probability': 'High',
                'urgency': 'Moderate',
                'supporting_evidence': [f"Bilirubin {liver['Total_Bilirubin']} mg/dL"],
                'next_step': 'Hepatitis serologies, abdominal ultrasound, INR, albumin'
            })
    
    return sorted(diagnoses, key=lambda x: {'Critical': 0, 'High': 1, 'Moderate': 2, 'Low': 3}.get(x['urgency'], 4))

def generate_recommendations(categorized_tests: Dict, diagnoses: List[Dict]) -> List[str]:
    """Generate next step recommendations"""
    recommendations = []
    
    # Critical value protocols
    for cat, tests in categorized_tests.items():
        for test, value in tests.items():
            if test in CRITICAL_VALUES and isinstance(value, (int, float)):
                low, high = CRITICAL_VALUES[test]
                if value < low or value > high:
                    recommendations.append(f"🚨 URGENT: Critical {test} value ({value}) - immediate physician notification required")
    
    # Diagnosis-specific recommendations
    for dx in diagnoses:
        recommendations.append(f"📋 {dx['next_step']}")
    
    # Category-specific follow-up
    if 'Hematology' in categorized_tests:
        recommendations.append("🔬 Peripheral blood smear review if not already performed")
    
    if 'Liver_Function' in categorized_tests:
        recommendations.append("🔬 Consider abdominal imaging if liver enzymes elevated >3x ULN")
    
    if 'Kidney_Function' in categorized_tests:
        recommendations.append("🔬 Monitor electrolytes closely if eGFR <60")
    
    return list(dict.fromkeys(recommendations))  # Remove duplicates

def generate_comprehensive_analysis(categorized_tests: Dict, gender: str, age: int) -> Dict:
    """Generate comprehensive analysis"""
    analysis = {
        'summary': [],
        'categories': {},
        'diagnoses': [],
        'next_steps': [],
        'critical_alerts': []
    }
    
    # Check critical values
    all_values = {}
    for cat_tests in categorized_tests.values():
        all_values.update(cat_tests)
    
    criticals = check_critical_values(all_values)
    if criticals:
        analysis['critical_alerts'] = criticals
    
    # Category-specific analysis
    category_analyzers = {
        'Hematology': analyze_hematology_patterns,
        'Liver_Function': analyze_liver_patterns,
        'Kidney_Function': analyze_kidney_patterns,
        'Metabolic': analyze_metabolic_patterns,
        'Endocrine': analyze_thyroid_patterns,
        'Lipid_Profile': analyze_lipid_patterns,
        'Immunology_Rheumatology': analyze_rheumatology_patterns
    }
    
    for category, tests in categorized_tests.items():
        if not tests or category not in category_analyzers:
            continue
        
        patterns = category_analyzers[category](tests)
        
        # Count abnormalities
        abnormalities = []
        for test, value in tests.items():
            if isinstance(value, (int, float)) and test in REFERENCE_RANGES:
                ref = REFERENCE_RANGES[test]
                if 'male' in ref and 'female' in ref:
                    low, high = ref[gender]
                else:
                    low, high = ref['range']
                
                if value < low or value > high:
                    abnormalities.append({
                        'test': test,
                        'value': value,
                        'direction': 'low' if value < low else 'high'
                    })
        
        analysis['categories'][category] = {
            'patterns': patterns,
            'abnormalities': abnormalities
        }
        
        if abnormalities:
            analysis['summary'].append(f"{category.replace('_', ' ')}: {len(abnormalities)} abnormal parameters")
    
    # Cross-category analysis
    analysis['diagnoses'] = generate_differential_diagnosis(categorized_tests, gender, age)
    analysis['next_steps'] = generate_recommendations(categorized_tests, analysis['diagnoses'])
    
    # RAG enhancement if available
    if rag_system and hasattr(rag_system, 'enhance_analysis'):
        try:
            rag_insights = rag_system.enhance_analysis(categorized_tests, analysis)
            analysis['rag_insights'] = rag_insights
        except:
            analysis['rag_insights'] = "RAG analysis temporarily unavailable"
    else:
        analysis['rag_insights'] = "RAG system not initialized - running rule-based analysis only"
    
    return analysis

def main():
    st.markdown('<h1 class="main-header">🧬 MedLab AI Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive Blood Investigation Analysis with AI-Powered Intelligence</p>', unsafe_allow_html=True)
    
    # RAG Status indicator
    if rag_system:
        st.markdown('<div class="rag-status">🧠 RAG Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rag-status" style="background: #f59e0b;">⚡ Basic Mode</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Patient Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        
        st.header("Data Input")
        input_method = st.radio("Input Method", ["Upload Document", "Manual Entry"])
        
        if input_method == "Upload Document":
            uploaded_file = st.file_uploader("Upload Lab Report", 
                                           type=['pdf', 'png', 'jpg', 'jpeg'])
            
            if uploaded_file and st.button("🔍 Extract Data"):
                with st.spinner("Processing document with OCR..."):
                    text = extract_text_from_document(uploaded_file)
                    if text:
                        parsed = parse_lab_values(text)
                        st.session_state.parsed_values.update(parsed)
                        st.success(f"Extracted {len(parsed)} parameters")
        
        st.header("Analysis Options")
        analysis_depth = st.select_slider("Analysis Depth", 
                                        options=["Screening", "Standard", "Comprehensive", "Academic"])
        generate_report = st.button("📊 Generate Full Report")
    
    # Main content area
    if st.session_state.parsed_values:
        st.markdown("---")
        
        # Tabs for organization
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Review & Correct", "🔬 Analysis", "🩺 Diagnoses", "📑 Report"])
        
        with tab1:
            st.subheader("Extracted Values - Review and Correct")
            st.markdown("*Verify automatically extracted values and make corrections if needed*")
            
            categorized = categorize_tests(st.session_state.parsed_values)
            
            for category, tests in categorized.items():
                if tests:
                    with st.expander(f"{category.replace('_', ' ')} ({len(tests)} parameters)", expanded=True):
                        for test, value in list(tests.items()):
                            display_parameter_card(test, value, category, gender.lower(), editable=True)
            
            if st.button("➕ Add Missing Parameter"):
                st.session_state.correction_mode = True
            
            if st.session_state.correction_mode:
                with st.form("add_parameter"):
                    cols = st.columns(3)
                    with cols[0]:
                        new_test = st.selectbox("Parameter", list(REFERENCE_RANGES.keys()))
                    with cols[1]:
                        new_value = st.number_input("Value", step=0.01)
                    with cols[2]:
                        if st.form_submit_button("Add"):
                            st.session_state.parsed_values[new_test] = new_value
                            st.session_state.correction_mode = False
                            st.rerun()
        
        with tab2:
            if categorized:
                st.subheader("Category-Based Analysis")
                
                # Run comprehensive analysis
                analysis = generate_comprehensive_analysis(categorized, gender.lower(), age)
                
                # Display critical alerts first
                if analysis['critical_alerts']:
                    st.error("🚨 CRITICAL VALUES DETECTED")
                    for alert in analysis['critical_alerts']:
                        st.markdown(f"""
                        <div style="background-color: #fee2e2; border: 2px solid #dc2626; padding: 15px; border-radius: 8px; margin: 10px 0;">
                            <strong>{alert['test']}</strong>: {alert['value']} (Critical range: {alert['range']})<br>
                            <em>Immediate action required</em>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Category analysis
                for category, cat_analysis in analysis['categories'].items():
                    if cat_analysis['abnormalities'] or cat_analysis['patterns']:
                        icon = "🔴" if cat_analysis['abnormalities'] else "🟡"
                        with st.expander(f"{icon} {category.replace('_', ' ')}", expanded=True):
                            
                            if cat_analysis['patterns']:
                                st.markdown("**Recognized Patterns:**")
                                for pattern in cat_analysis['patterns']:
                                    st.markdown(f"- {pattern}")
                            
                            if cat_analysis['abnormalities']:
                                st.markdown("**Abnormal Parameters:**")
                                for abnorm in cat_analysis['abnormalities']:
                                    st.markdown(f"- **{abnorm['test']}**: {abnorm['value']} ({abnorm['direction']})")
                
                # RAG insights
                if 'rag_insights' in analysis:
                    with st.expander("🧠 AI-Enhanced Insights", expanded=True):
                        st.markdown(analysis['rag_insights'])
        
        with tab3:
            if analysis['diagnoses']:
                st.subheader("Differential Diagnoses")
                
                for i, dx in enumerate(analysis['diagnoses']):
                    urgency_colors = {
                        'Critical': '#dc2626',
                        'High': '#ea580c',
                        'Moderate': '#ca8a04',
                        'Low': '#16a34a'
                    }
                    color = urgency_colors.get(dx['urgency'], '#6b7280')
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {color}; background-color: #f9fafb; padding: 20px; margin: 15px 0; border-radius: 8px;">
                        <h4 style="color: {color}; margin-top: 0;">{i+1}. {dx['condition']} 
                        <span style="font-size: 0.8em; background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px;">{dx['urgency']}</span></h4>
                        <p><strong>Probability:</strong> {dx['probability']}</p>
                        <p><strong>Supporting Evidence:</strong> {', '.join(dx['supporting_evidence'])}</p>
                        <p><strong>Next Steps:</strong> {dx['next_step']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab4:
            if generate_report:
                st.subheader("Comprehensive Laboratory Report")
                
                report_data = {
                    'patient_info': {'gender': gender, 'age': age, 'date': datetime.now().strftime('%Y-%m-%d')},
                    'results': st.session_state.parsed_values,
                    'analysis': analysis
                }
                
                report_json = json.dumps(report_data, indent=2)
                st.download_button(
                    label="📥 Download Report (JSON)",
                    data=report_json,
                    file_name=f"lab_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                st.markdown("### Executive Summary")
                st.write(f"Total parameters analyzed: {len(st.session_state.parsed_values)}")
                st.write(f"Abnormal findings: {sum(len(cat.get('abnormalities', [])) for cat in analysis['categories'].values())}")
                st.write(f"Critical alerts: {len(analysis['critical_alerts'])}")
                st.write(f"Potential diagnoses identified: {len(analysis['diagnoses'])}")

    else:
        st.info("👆 Upload a lab report or enter values manually to begin analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="category-card">
                <h3>🩸 Hematology</h3>
                <p>CBC, coagulation, blood cancer screening</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="category-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>🔥 Metabolism</h3>
                <p>Diabetes, lipids, metabolic syndrome</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="category-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>🦋 Endocrine</h3>
                <p>Thyroid, adrenal, pituitary function</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
