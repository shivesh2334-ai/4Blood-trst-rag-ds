# rag_components.py - FIXED VERSION
import os
import json
from typing import Dict, List, Any

# FIXED: Updated imports for newer langchain versions - using only langchain_community
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    HuggingFaceEmbeddings = None
    FAISS = None

# Handle text splitter separately
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

# Handle Document class
try:
    from langchain.schema import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        Document = None

class MedLabRAG:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.initialized = False
        
        # Only initialize if all imports are available
        if HuggingFaceEmbeddings is None or FAISS is None or RecursiveCharacterTextSplitter is None:
            print("Required LangChain components not available - RAG features disabled")
            return
            
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
                try:
                    self.vectorstore = FAISS.load_local("medical_vectorstore", self.embeddings)
                except Exception as e:
                    print(f"Could not load existing vectorstore: {e}")
                    self._create_knowledge_base()
            else:
                self._create_knowledge_base()
            
            self.initialized = True
        except Exception as e:
            print(f"RAG initialization error: {e}")
            self.initialized = False
    
    def _create_knowledge_base(self):
        """Create medical knowledge base from structured data"""
        medical_texts = self._load_medical_knowledge()
        
        if not medical_texts or RecursiveCharacterTextSplitter is None or Document is None:
            return
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Convert texts to Document objects
        documents = [Document(page_content=text, metadata={"source": "medical_knowledge"}) 
                    for text in medical_texts]
        
        chunks = text_splitter.split_documents(documents)
        
        if chunks and self.embeddings and FAISS:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            try:
                self.vectorstore.save_local("medical_vectorstore")
            except Exception as e:
                print(f"Could not save vectorstore: {e}")
    
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
            
            # Metabolic Knowledge
            """
            Diabetes Mellitus Type 2: HbA1c ≥6.5%, fasting glucose ≥126 mg/dL, 
            random glucose ≥200 with symptoms. Insulin resistance, metabolic syndrome. 
            Complications: retinopathy, nephropathy, neuropathy, cardiovascular disease. 
            Next steps: Ophthalmology, urine microalbumin, lipid panel, ACE inhibitor.
            """,
            
            # Liver Knowledge
            """
            Acute Hepatitis: ALT > AST, both elevated >10x ULN. Viral (A, B, C, E), 
            drug-induced, autoimmune, ischemic. Jaundice, dark urine, pale stools. 
            Next steps: Viral serologies, autoimmune markers (ANA, SMA, LKM), 
            drug history, abdominal ultrasound.
            """,
            
            # Kidney Knowledge
            """
            Acute Kidney Injury: Rise in creatinine by 0.3 mg/dL in 48h or 1.5x baseline 
            in 7 days. Prerenal (BUN:Cr >20), intrinsic (ATN, GN, AIN), postrenal. 
            Next steps: Urinalysis, renal ultrasound, fluid challenge, 
            stop nephrotoxins, nephrology if severe.
            """,
            
            # Thyroid Knowledge
            """
            Hashimoto's Thyroiditis: Elevated TSH, low/normal FT4, positive anti-TPO 
            and/or anti-thyroglobulin. Goiter, hypothyroidism. Most common cause of 
            hypothyroidism in iodine-sufficient areas. 
            Next steps: Levothyroxine replacement, monitor TSH annually.
            """,
            
            # Rheumatology Knowledge
            """
            Rheumatoid Arthritis: Symmetric polyarthritis, morning stiffness >1 hour, 
            positive RF and/or anti-CCP, elevated ESR/CRP. Erosions on X-ray. 
            Next steps: Methotrexate first-line, DMARDs, biologics if inadequate response.
            """
        ]
        
        return knowledge_base
    
    def enhance_analysis(self, categorized_tests: Dict, rule_based_analysis: Dict) -> str:
        """Enhance analysis with RAG-retrieved knowledge"""
        if not self.initialized or not self.vectorstore:
            return "RAG system not available. Using rule-based analysis only."
        
        try:
            # Build query from abnormal findings
            query_parts = []
            for category, tests in categorized_tests.items():
                for test, value in tests.items():
                    if isinstance(value, (int, float)):
                        # Simple threshold check
                        if test in ['Hemoglobin', 'WBC', 'Platelets', 'Glucose_Fasting', 'HbA1c', 'Creatinine', 'TSH']:
                            query_parts.append(f"{test} {value}")
            
            if not query_parts:
                return "All parameters within normal limits. No additional insights needed."
            
            query = "Laboratory abnormalities: " + ", ".join(query_parts[:5])
            
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
            """
            
            return insights
            
        except Exception as e:
            return f"RAG enhancement error: {str(e)}. Proceeding with standard analysis."
    
    def query_knowledge_base(self, question: str) -> str:
        """Allow direct querying of medical knowledge base"""
        if not self.initialized or not self.vectorstore:
            return "Knowledge base not available"
        
        try:
            docs = self.vectorstore.similarity_search(question, k=2)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Query error: {str(e)}"
