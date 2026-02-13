# rag_components.py
# RAG implementation for medical laboratory analysis

import os
import json
from typing import Dict, List, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np

class MedLabRAG:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.initialized = False
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize embeddings and vector store"""
        try:
            # Use lightweight embeddings suitable for medical text
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Create or load vector store
            if os.path.exists("medical_vectorstore"):
                self.vectorstore = FAISS.load_local("medical_vectorstore", self.embeddings)
            else:
                self._create_knowledge_base()
            
            self.initialized = True
        except Exception as e:
            print(f"RAG initialization error: {e}")
            self.initialized = False
    
    def _create_knowledge_base(self):
        """Create medical knowledge base from structured data"""
        medical_texts = self._load_medical_knowledge()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.create_documents(medical_texts)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local("medical_vectorstore")
    
    def _load_medical_knowledge(self) -> List[str]:
        """Load comprehensive medical knowledge for lab interpretation"""
        knowledge_base = [
            # Hematology Knowledge
            """
            Iron Deficiency Anemia: Characterized by low MCV (<80 fL), high RDW (>14.5%), 
            low ferritin (<15 ng/mL), low iron, high TIBC. Causes include chronic blood loss, 
            poor intake, malabsorption. Next steps: Iron studies, stool occult blood, endoscopy 
            if GI source suspected.
            """,
            
            """
            Vitamin B12 Deficiency: Macrocytic anemia (MCV >100 fL), hypersegmented neutrophils, 
            low B12 (<200 pg/mL), elevated methylmalonic acid. Causes: pernicious anemia, 
            gastrectomy, ileal disease, vegan diet. Neurological symptoms may precede anemia.
            Next steps: Intrinsic factor antibodies, Schilling test, neurology referral.
            """,
            
            """
            Folate Deficiency: Macrocytic anemia, low folate (<2 ng/mL), normal B12. 
            Common in alcoholism, pregnancy, hemolysis, methotrexate use. 
            No neurological symptoms unlike B12 deficiency. 
            Next steps: Dietary assessment, alcohol history, medication review.
            """,
            
            """
            Hemolytic Anemia: High LDH, high indirect bilirubin, low haptoglobin, 
            high reticulocyte count. Peripheral smear shows spherocytes, schistocytes, 
            or bite cells depending on cause. 
            Next steps: Direct/indirect Coombs test, hemoglobin electrophoresis, G6PD screen.
            """,
            
            """
            Acute Leukemia: Blasts >20% in peripheral blood or bone marrow. 
            Pancytopenia common. Auer rods in AML, lymphoblasts in ALL. 
            Symptoms: fatigue, infections, bleeding. 
            Next steps: Urgent hematology, bone marrow biopsy, flow cytometry, cytogenetics.
            """,
            
            """
            Chronic Lymphocytic Leukemia (CLL): Lymphocytosis >5000/μL with smudge cells, 
            CD5+, CD19+, CD23+ immunophenotype. Often asymptomatic in elderly. 
            Rai and Binet staging systems. 
            Next steps: Flow cytometry, FISH for del(13q), TP53 mutation.
            """,
            
            """
            Multiple Myeloma: CRAB features (Calcium elevated, Renal failure, Anemia, Bone lesions). 
            Monoclonal spike on protein electrophoresis, light chains in urine. 
            Next steps: Serum protein electrophoresis, free light chains, skeletal survey, 
            bone marrow biopsy.
            """,
            
            # Metabolic Knowledge
            """
            Diabetes Mellitus Type 2: HbA1c ≥6.5%, fasting glucose ≥126 mg/dL, 
            random glucose ≥200 with symptoms. Insulin resistance, metabolic syndrome. 
            Complications: retinopathy, nephropathy, neuropathy, cardiovascular disease. 
            Next steps: Ophthalmology, urine microalbumin, lipid panel, ACE inhibitor.
            """,
            
            """
            Diabetic Ketoacidosis (DKA): Glucose >250, pH <7.3, bicarbonate <18, 
            ketonemia/ketonuria. Medical emergency. Precipitated by infection, 
            insulin omission, MI, stroke. 
            Next steps: ICU admission, insulin drip, fluid resuscitation, electrolyte monitoring.
            """,
            
            """
            Metabolic Syndrome: 3 of 5 criteria: waist >102cm (M) or >88cm (F), 
            triglycerides ≥150, HDL <40 (M) or <50 (F), BP ≥130/85, glucose ≥100. 
            Increased cardiovascular risk. 
            Next steps: Lifestyle modification, statin therapy, BP control, glucose monitoring.
            """,
            
            # Liver Knowledge
            """
            Acute Hepatitis: ALT > AST, both elevated >10x ULN. Viral (A, B, C, E), 
            drug-induced, autoimmune, ischemic. Jaundice, dark urine, pale stools. 
            Next steps: Viral serologies, autoimmune markers (ANA, SMA, LKM), 
            drug history, abdominal ultrasound.
            """,
            
            """
            Alcoholic Liver Disease: AST:ALT ratio >2:1, both elevated <300 U/L, 
            elevated GGT, macrocytosis. History of heavy alcohol use. 
            Spectrum: fatty liver, alcoholic hepatitis, cirrhosis. 
            Next steps: Abstinence support, nutrition, prednisolone if severe hepatitis.
            """,
            
            """
            Primary Biliary Cholangitis: ALP >1.5x ULN, elevated GGT, positive AMA, 
            elevated IgM. Middle-aged women, pruritus, fatigue. 
            Next steps: Liver biopsy if AMA negative, ursodeoxycholic acid, 
            screen for osteoporosis.
            """,
            
            """
            Acute Liver Failure: INR >1.5, encephalopathy, no cirrhosis, 
            illness <26 weeks. Etiologies: acetaminophen, viral hepatitis, 
            autoimmune, Wilson disease. 
            Next steps: ICU, NAC for acetaminophen, transplant evaluation, 
            lactulose/rifaximin for encephalopathy.
            """,
            
            # Kidney Knowledge
            """
            Acute Kidney Injury: Rise in creatinine by 0.3 mg/dL in 48h or 1.5x baseline 
            in 7 days. Prerenal (BUN:Cr >20), intrinsic (ATN, GN, AIN), postrenal. 
            Next steps: Urinalysis, renal ultrasound, fluid challenge, 
            stop nephrotoxins, nephrology if severe.
            """,
            
            """
            Chronic Kidney Disease: eGFR <60 for >3 months or markers of kidney damage 
            (albuminuria, hematuria, structural abnormalities). Stages G1-G5 based on eGFR. 
            Next steps: BP control (ACEi/ARB), SGLT2i, treat complications 
            (anemia, bone disease, acidosis), nephrology referral if G4-G5.
            """,
            
            """
            Nephrotic Syndrome: Proteinuria >3.5g/day, hypoalbuminemia <3g/dL, 
            edema, hyperlipidemia. Causes: minimal change, FSGS, membranous, diabetes, amyloid. 
            Next steps: Renal biopsy if adult, lipid management, anticoagulation if indicated.
            """,
            
            """
            Nephritic Syndrome: Hematuria, proteinuria <3.5g/day, hypertension, 
            edema, renal insufficiency. Post-infectious GN, IgA nephropathy, 
            lupus nephritis, vasculitis. 
            Next steps: Complement levels, ANA, ANCA, anti-GBM, renal biopsy.
            """,
            
            # Thyroid Knowledge
            """
            Hashimoto's Thyroiditis: Elevated TSH, low/normal FT4, positive anti-TPO 
            and/or anti-thyroglobulin. Goiter, hypothyroidism. Most common cause of 
            hypothyroidism in iodine-sufficient areas. 
            Next steps: Levothyroxine replacement, monitor TSH annually.
            """,
            
            """
            Graves' Disease: Suppressed TSH, elevated FT4/FT3, positive TRAb, 
            goiter, ophthalmopathy. Autoimmune hyperthyroidism. 
            Next steps: Methimazole or PTU, beta-blocker, radioactive iodine or 
            surgery if relapse.
            """,
            
            """
            Subclinical Hypothyroidism: TSH 4.5-10, normal FT4. Treat if TSH >10, 
            positive antibodies, symptoms, pregnancy desire, or cardiovascular risk. 
            Next steps: Repeat in 6 months, anti-TPO, consider trial of levothyroxine.
            """,
            
            # Rheumatology Knowledge
            """
            Rheumatoid Arthritis: Symmetric polyarthritis, morning stiffness >1 hour, 
            positive RF and/or anti-CCP, elevated ESR/CRP. Erosions on X-ray. 
            Next steps: Methotrexate first-line, DMARDs, biologics if inadequate response.
            """,
            
            """
            Systemic Lupus Erythematosus: Multi-system involvement, positive ANA, 
            anti-dsDNA, low complement, malar rash, photosensitivity, oral ulcers, 
            arthritis, serositis, renal involvement. 
            Next steps: Disease activity assessment, hydroxychloroquine, 
            immunosuppression based on severity.
            """,
            
            """
            ANCA-Associated Vasculitis: Granulomatosis with polyangiitis, 
            microscopic polyangiitis, eosinophilic granulomatosis with polyangiitis. 
            c-ANCA (PR3) or p-ANCA (MPO) positive. 
            Next steps: High-dose steroids, cyclophosphamide or rituximab, 
            PJP prophylaxis.
            """,
            
            """
            Polymyalgia Rheumatica: Age >50, shoulder/hip girdle pain and stiffness, 
            elevated ESR/CRP, rapid response to steroids. Associated with giant cell arteritis. 
            Next steps: Prednisone 15-20mg, temporal artery biopsy if headache/visual symptoms.
            """,
            
            # Coagulation Knowledge
            """
            Disseminated Intravascular Coagulation (DIC): Prolonged PT/aPTT, 
            low fibrinogen, elevated D-dimer, thrombocytopenia, schistocytes. 
            Causes: sepsis, trauma, malignancy, obstetric complications. 
            Next steps: Treat underlying cause, blood product support, heparin if thrombosis.
            """,
            
            """
            Immune Thrombocytopenia (ITP): Isolated thrombocytopenia <100,000, 
            normal or increased MPV, no splenomegaly, exclusion of other causes. 
            Next steps: Steroids first-line, IVIG if bleeding, TPO agonists, splenectomy.
            """,
            
            """
            Heparin-Induced Thrombocytopenia (HIT): Platelet count falls >50% 
            from baseline 5-10 days after heparin exposure, thrombosis, 
            positive HIT antibody (PF4). 
            Next steps: Stop all heparin, argatroban or fondaparinux, 
            non-heparin anticoagulation for 3 months if thrombosis.
            """,
            
            # Oncology Knowledge
            """
            Tumor Marker Interpretation: AFP for hepatocellular carcinoma and 
            non-seminomatous germ cell tumors. CEA for colorectal cancer monitoring. 
            CA-125 for ovarian cancer (elevated in many benign conditions). 
            PSA for prostate cancer screening (controversial). 
            Next steps: Imaging confirmation, not diagnostic alone.
            """,
            
            # Emergency/Critical Care
            """
            Sepsis and Septic Shock: Suspected infection with SOFA score ≥2, 
            lactate >2 mmol/L, vasopressor requirement. 
            Next steps: Blood cultures before antibiotics, broad-spectrum antibiotics 
            within 1 hour, fluid resuscitation, source control, ICU admission.
            """,
            
            """
            Acute Coronary Syndrome: Elevated troponin, ST changes, chest pain. 
            STEMI, NSTEMI, unstable angina. 
            Next steps: Aspirin, P2Y12 inhibitor, anticoagulation, statin, 
            beta-blocker, revascularization.
            """,
            
            """
            Pulmonary Embolism: Dyspnea, tachypnea, hypoxia, elevated D-dimer, 
            CT pulmonary angiography showing filling defect. 
            Next steps: Anticoagulation (heparin, DOAC), thrombolysis if massive, 
            IVC filter if contraindicated.
            """,
        ]
        
        return knowledge_base
    
    def enhance_analysis(self, categorized_tests: Dict, rule_based_analysis: Dict) -> str:
        """Enhance analysis with RAG-retrieved knowledge"""
        if not self.initialized:
            return "RAG system not available. Using rule-based analysis only."
        
        try:
            # Build query from abnormal findings
            query_parts = []
            for category, tests in categorized_tests.items():
                for test, value in tests.items():
                    if isinstance(value, (int, float)) and test in self._get_reference_range(test):
                        ref = self._get_reference_range(test)
                        if 'range' in ref:
                            low, high = ref['range']
                        else:
                            low, high = ref.get('male', ref.get('female', (0, 0)))
                        
                        if value < low or value > high:
                            query_parts.append(f"{test} {value} ({'low' if value < low else 'high'})")
            
            if not query_parts:
                return "All parameters within normal limits. No additional insights needed."
            
            query = "Laboratory abnormalities: " + ", ".join(query_parts[:5])  # Limit to top 5
            
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate enhanced insights
            insights = f"""
            **AI-Enhanced Clinical Insights:**
            
            Based on pattern recognition and medical literature:
            
            {context[:1000]}...
            
            **Key Considerations:**
            1. Correlation with clinical presentation is essential
            2. Trend analysis provides more value than single measurements
            3. Consider pre-analytical variables (fasting, medications, hemolysis)
            4. Age, sex, and ethnicity-specific reference ranges may apply
            
            **Literature-Based Recommendations:**
            - {self._generate_specific_recommendations(query_parts)}
            """
            
            return insights
            
        except Exception as e:
            return f"RAG enhancement error: {str(e)}. Proceeding with standard analysis."
    
    def _get_reference_range(self, test: str) -> Dict:
        """Get reference range for a test"""
        from medical_reference import REFERENCE_RANGES
        return REFERENCE_RANGES.get(test, {})
    
    def _generate_specific_recommendations(self, abnormalities: List[str]) -> str:
        """Generate specific recommendations based on abnormalities"""
        recommendations = []
        
        for abnorm in abnormalities:
            if 'HbA1c' in abnorm and 'high' in abnorm:
                recommendations.append("HbA1c elevation: Consider continuous glucose monitoring, endocrinology referral if >9%")
            elif 'Creatinine' in abnorm and 'high' in abnorm:
                recommendations.append("Creatinine elevation: Calculate eGFR trend, check for nephrotoxins, consider nephrology if >30% decline")
            elif 'TSH' in abnorm:
                recommendations.append("Thyroid dysfunction: Check free T4 to determine severity, anti-TPO if hypothyroidism")
            elif 'Platelets' in abnorm and 'low' in abnorm:
                recommendations.append("Thrombocytopenia: Peripheral smear essential, rule out pseudothrombocytopenia with citrate tube")
        
        return "\n- ".join(recommendations) if recommendations else "Continue routine monitoring"
    
    def query_knowledge_base(self, question: str) -> str:
        """Allow direct querying of medical knowledge base"""
        if not self.initialized:
            return "Knowledge base not available"
        
        try:
            docs = self.vectorstore.similarity_search(question, k=2)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Query error: {str(e)}"

# Initialize singleton
_rag_instance = None

def get_rag_instance():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MedLabRAG()
    return _rag_instance
      
