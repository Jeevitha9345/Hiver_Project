import os
import sys
sys.path.insert(0, os.path.abspath("."))

from typing import Dict, Any, Optional

from src.intent_classifier import SentenceTransformerIntentClassifier
from src.retriever import HistoricalSupportRetriever
from src.escalation import EscalationPolicy
from src.response_generator import GroundedResponseGenerator
from src.preprocessing import TextPreprocessor

class HiverSupportAgent:
    """
    Complete, production-grade Customer Support AI Agent.
    Coordinates intent classification, historical evidence retrieval,
    safety escalation policy, and grounded response synthesis.
    """
    
    def __init__(
        self,
        classifier_model_path: Optional[str] = None,
        retriever_index_path: Optional[str] = None,
        retriever_metadata_path: Optional[str] = None
    ):
        print("[Agent] Initializing Hiver Customer Support Agent...")
        self.preprocessor = TextPreprocessor()
        
        if classifier_model_path and os.path.exists(classifier_model_path):
            self.classifier = SentenceTransformerIntentClassifier.load(classifier_model_path)
        else:
            self.classifier = SentenceTransformerIntentClassifier.load()
            
        if retriever_index_path and retriever_metadata_path:
            self.retriever = HistoricalSupportRetriever.load(retriever_index_path, retriever_metadata_path)
        else:
            self.retriever = HistoricalSupportRetriever.load()
            
        self.escalation_policy = EscalationPolicy(min_retrieval_sim=0.40, min_intent_conf=0.35)
        self.generator = GroundedResponseGenerator()
        print("[Agent] Hiver Customer Support Agent ready.")

    def process_message(self, message: str, top_k_retrieval: int = 3) -> Dict[str, Any]:
        cleaned_msg = self.preprocessor.clean_customer_query(message)
        
        intent_res = self.classifier.predict(cleaned_msg)
        intent_label = intent_res["intent"]
        intent_confidence = intent_res["confidence"]
        
        retrieved_examples = self.retriever.retrieve(cleaned_msg, top_k=top_k_retrieval)
        
        escalation_res = self.escalation_policy.evaluate(
            message=cleaned_msg,
            intent=intent_label,
            intent_confidence=intent_confidence,
            retrieval_results=retrieved_examples
        )
        
        reply_res = self.generator.generate(
            customer_message=cleaned_msg,
            predicted_intent=intent_label,
            retrieved_examples=retrieved_examples,
            escalation=escalation_res
        )
        
        overall_confidence = round((intent_confidence * 0.4) + (escalation_res["confidence"] * 0.4) + (reply_res["confidence"] * 0.2), 2)
        
        return {
            "message": message,
            "cleaned_message": cleaned_msg,
            "intent": {
                "label": intent_label,
                "confidence": intent_confidence,
                "probabilities": intent_res.get("probabilities", {})
            },
            "retrieved_examples": retrieved_examples,
            "decision": escalation_res["decision"],
            "decision_reason": escalation_res["reason"],
            "escalation_trigger": escalation_res.get("trigger", "N/A"),
            "reply": reply_res["draft_reply"],
            "evidence": reply_res.get("evidence", []),
            "warnings": reply_res.get("warnings", []),
            "confidence": overall_confidence
        }

if __name__ == "__main__":
    agent = HiverSupportAgent()
    sample_queries = [
        "Where is my package? The tracking has not updated for 3 days.",
        "Your representative was rude and hung up on me. Transfer me to a manager right now!",
        "Someone hacked into my account and ordered items with my card!",
        "When will the PlayStation 5 console be back in stock?"
    ]
    for q in sample_queries:
        print("\n" + "="*60)
        print(f"QUERY: {q}")
        res = agent.process_message(q)
        print(f"INTENT: {res['intent']['label']} ({res['intent']['confidence']:.2f})")
        print(f"DECISION: {res['decision']} (Reason: {res['decision_reason']})")
        print(f"REPLY: {res['reply']}")
