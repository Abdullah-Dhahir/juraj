import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

# Updated imports for LangChain compatibility
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
except ImportError:
    # Fallback for older versions
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document

logger = logging.getLogger(__name__)

class NLPService:
    """NLP service with RAG pipeline for Karlsruhe public services"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize NLP service with RAG pipeline
        
        Args:
            model_path: Path to local LLM model (e.g., Mistral GGUF)
        """
        self.model_name = "mistral-7b-instruct"
        self.model_path = model_path or "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
        # Initialize components
        self._load_embeddings()
        self._load_knowledge_base()
        self._load_llm()
        self._setup_qa_chain()
    
    def _load_embeddings(self):
        """Load embedding model for vector search"""
        try:
            logger.info("🔄 Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}  # Use GPU if available
            )
            logger.info("✅ Embedding model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            raise
    
    def _load_knowledge_base(self):
        """Load and vectorize knowledge base about Karlsruhe public services"""
        try:
            logger.info("🔄 Loading knowledge base...")
            
            # Sample knowledge base for Karlsruhe public services
            knowledge_data = self._get_karlsruhe_knowledge()
            
            # Create documents
            documents = []
            for item in knowledge_data:
                doc = Document(
                    page_content=item["content"],
                    metadata={"source": item["source"], "category": item["category"]}
                )
                documents.append(doc)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            logger.info(f"✅ Knowledge base loaded with {len(split_docs)} chunks!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge base: {e}")
            raise
    
    def _load_llm(self):
        """Load local LLM model"""
        try:
            logger.info(f"🔄 Loading LLM model from {self.model_path}...")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"⚠️ Model file not found: {self.model_path}")
                logger.info("🔄 Using fallback simple response generation...")
                self.llm = None
                return
            
            self.llm = LlamaCpp(
                model_path=self.model_path,
                temperature=0.3,
                max_tokens=512,
                top_p=1,
                verbose=False,
                n_ctx=2048
            )
            
            logger.info("✅ LLM model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load LLM: {e}")
            logger.info("🔄 Using fallback simple response generation...")
            self.llm = None
    
    def _setup_qa_chain(self):
        """Setup QA chain with custom prompt"""
        try:
            if not self.llm or not self.vectorstore:
                logger.info("🔄 Setting up simple QA without LLM...")
                return
            
            # Custom prompt template for German/English responses
            prompt_template = """Du bist ein hilfreicher Assistent für öffentliche Dienstleistungen in Karlsruhe. 
            Beantworte die Frage basierend auf dem gegebenen Kontext. Wenn du die Antwort nicht weißt, sage ehrlich, dass du es nicht weißt.

            Kontext: {context}

            Frage: {question}

            Antwort:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            logger.info("✅ QA chain setup completed!")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup QA chain: {e}")
            self.qa_chain = None
    
    async def generate_response(self, message: str) -> str:
        """
        Generate response to user message using RAG pipeline
        
        Args:
            message: User input message
            
        Returns:
            Generated response
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self._generate_response_sync, 
                message
            )
            return response
            
        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return "Entschuldigung, ich hatte ein Problem bei der Verarbeitung Ihrer Anfrage. Können Sie es bitte noch einmal versuchen?"
    
    def _generate_response_sync(self, message: str) -> str:
        """Synchronous response generation"""
        try:
            # If we have full RAG pipeline
            if self.qa_chain:
                result = self.qa_chain({"query": message})
                return result["result"].strip()
            
            # Fallback: Simple retrieval without LLM
            elif self.vectorstore:
                docs = self.vectorstore.similarity_search(message, k=2)
                if docs:
                    # Simple template-based response
                    context = docs[0].page_content
                    return self._simple_response_template(message, context)
                else:
                    return self._default_response(message)
            
            # Ultimate fallback
            else:
                return self._default_response(message)
                
        except Exception as e:
            logger.error(f"❌ Sync response generation error: {e}")
            return "Es tut mir leid, ich konnte Ihre Anfrage nicht verarbeiten."
    
    def _simple_response_template(self, question: str, context: str) -> str:
        """Simple template-based response when LLM is not available"""
        return f"Basierend auf den verfügbaren Informationen zu Karlsruhe: {context[:200]}... Benötigen Sie weitere Details zu diesem Thema?"
    
    def _default_response(self, message: str) -> str:
        """Default response when no context is found"""
        if any(word in message.lower() for word in ["hallo", "hi", "guten tag", "hello"]):
            return "Hallo! Ich helfe Ihnen gerne bei Fragen zu öffentlichen Dienstleistungen in Karlsruhe. Wie kann ich Ihnen behilflich sein?"
        
        elif any(word in message.lower() for word in ["danke", "vielen dank", "thank you"]):
            return "Gerne! Falls Sie weitere Fragen haben, stehe ich Ihnen zur Verfügung."
        
        else:
            return "Entschuldigung, ich habe keine spezifischen Informationen zu Ihrer Anfrage. Können Sie Ihre Frage präzisieren oder sich an das Bürgerbüro Karlsruhe wenden?"
    
    def _get_karlsruhe_knowledge(self) -> List[Dict]:
        """Sample knowledge base for Karlsruhe public services"""
        return [
            {
                "content": "Das Bürgerbüro Karlsruhe befindet sich im Rathaus am Marktplatz. Öffnungszeiten: Mo-Fr 8:00-18:00, Sa 9:00-12:00. Hier können Sie Personalausweise, Reisepässe und Meldebescheinigungen beantragen.",
                "source": "karlsruhe.de",
                "category": "buergerbuero"
            },
            {
                "content": "Für die Anmeldung in Karlsruhe benötigen Sie: Personalausweis oder Reisepass, Wohnungsgeberbestätigung, bei verheirateten Personen die Heiratsurkunde. Die Anmeldung muss innerhalb von 14 Tagen nach Einzug erfolgen.",
                "source": "karlsruhe.de",
                "category": "anmeldung"
            },
            {
                "content": "Das Standesamt Karlsruhe führt Eheschließungen, Geburts- und Sterbeurkundungen durch. Terminvereinbarung erforderlich. Kontakt: standesamt@karlsruhe.de oder Tel. 0721/133-5301.",
                "source": "karlsruhe.de",
                "category": "standesamt"
            },
            {
                "content": "Karlsruher Verkehrsverbund (KVV): Einzelfahrkarten, Tageskarten und Monatskarten verfügbar. Studenten erhalten Ermäßigung mit Studienausweis. Online-Tickets über die KVV-App erhältlich.",
                "source": "kvv.de",
                "category": "verkehr"
            },
            {
                "content": "Abfallentsorgung in Karlsruhe: Restmüll alle 14 Tage, Bioabfall wöchentlich, Gelber Sack alle 14 Tage, Papier alle 4 Wochen. Sperrmüll nach Terminvereinbarung mit der Stadt Karlsruhe.",
                "source": "karlsruhe.de",
                "category": "abfall"
            },
            {
                "content": "Parkausweise für Anwohner können im Ordnungsamt beantragt werden. Kosten: 30€ pro Jahr. Benötigte Unterlagen: Fahrzeugschein, Personalausweis, Meldebescheinigung.",
                "source": "karlsruhe.de",
                "category": "parken"
            }
        ]
    
    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "has_llm": self.llm is not None,
            "has_vectorstore": self.vectorstore is not None,
            "has_qa_chain": self.qa_chain is not None,
            "status": "loaded"
        }