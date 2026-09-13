import re
from typing import Dict, Any, List, Optional

class EscalationPolicy:
    """
    Multi-layered decision engine for triage: AUTO_HANDLE vs ESCALATE.
    Combines deterministic safety rules, intent risk priors, retrieval quality,
    and classifier uncertainty.
    """
    
    # 1. Critical safety, legal & regulatory threat triggers
    LEGAL_REGULATORY_PATTERNS = [
        re.compile(r'\b(lawsuit|lawyer|attorney|sue|court|litigation)\b', re.IGNORECASE),
        re.compile(r'\b(ftc|bbb|better business bureau|consumer protection|regulatory|police|fir)\b', re.IGNORECASE),
        re.compile(r'\b(press|journalist|media|forbes|article|reporter|tiktok.*followers|viral)\b', re.IGNORECASE)
    ]
    
    # 2. Account compromise, identity fraud, and severe theft triggers
    SECURITY_FRAUD_PATTERNS = [
        re.compile(r'\b(hacked|compromised|stolen account|unauthorized order|fraud|identity theft)\b', re.IGNORECASE),
        re.compile(r'\b(scam|scammed|impersonation|remote access|anydesk|teamviewer)\b', re.IGNORECASE),
        re.compile(r'\b(unauthorized charge|unknown charge|card charged|deducted without|unauthorized.*prime|unauthorized.*card)\b', re.IGNORECASE),
        re.compile(r'\b(stolen from porch|stolen.*box|empty box.*phone|stolen unit|someone hacked)\b', re.IGNORECASE)
    ]
    
    # 3. Severe customer exasperation, repeated agent failures, demands for human
    HIGH_DISSATISFACTION_PATTERNS = [
        re.compile(r'\b(supervisor|manager|team lead|human agent|real human|transfer me|speak to a human)\b', re.IGNORECASE),
        re.compile(r'\b(rude|abrupt|disconnected|hung up|lied|promised.*lied|broken callback)\b', re.IGNORECASE),
        re.compile(r'\b(5th time|4th time|third time|3 weeks|2 weeks|weeks of emails|looping in circles|multiple agents)\b', re.IGNORECASE),
        re.compile(r'\b(worst customer service|pathetic|disgusted|stole my money|ruined.*birthday|ruined christmas|zero stars)\b', re.IGNORECASE)
    ]
    
    # 4. Physical safety or hazardous materials
    SAFETY_HAZARD_PATTERNS = [
        re.compile(r'\b(fire|smoking|smoke|exploded|hazardous|chemical leak|leaked acid|hazard)\b', re.IGNORECASE)
    ]
    
    # 5. Driver misconduct, carrier property damage, or driver disputes
    CARRIER_DISPUTE_PATTERNS = [
        re.compile(r'\b(driver.*(rain|threw|fence|past|neighbor|conduct|drove right past))\b', re.IGNORECASE),
        re.compile(r'\b(marked delivered.*(nothing|not received|nobody|empty)|handed to resident.*nobody)\b', re.IGNORECASE),
        re.compile(r'\b(guaranteed.*(birthday|hasn\'t even shipped)|label created for.*days|forwarded to different)\b', re.IGNORECASE),
        re.compile(r'\b(prime now.*(melt|perishable|ice cream)|different state|california instead of)\b', re.IGNORECASE)
    ]
    
    # 6. Critical billing discrepancies, duplicate charges, or SLA breaches
    BILLING_DISPUTE_PATTERNS = [
        re.compile(r'\b(only refunded|restocking fee|overdraft fee|bank.*tracer|tracer id|duplicate charge)\b', re.IGNORECASE),
        re.compile(r'\b(14 business days|10 days ago.*no progress|returned.*weeks ago.*no refund|never received.*proof)\b', re.IGNORECASE),
        re.compile(r'\b(refund.*different bank card|seller.*doubled|price has now doubled|dispute)\b', re.IGNORECASE)
    ]

    def __init__(self, min_retrieval_sim: float = 0.40, min_intent_conf: float = 0.35):
        self.min_retrieval_sim = min_retrieval_sim
        self.min_intent_conf = min_intent_conf

    def evaluate(
        self,
        message: str,
        intent: str,
        intent_confidence: float = 0.8,
        retrieval_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        msg = message.strip()
        
        # Rule 1: Legal / Regulatory / Media threats
        for pat in self.LEGAL_REGULATORY_PATTERNS:
            if pat.search(msg):
                return {
                    "decision": "ESCALATE",
                    "reason": "Legal, regulatory, or media escalation threat detected.",
                    "confidence": 0.99,
                    "trigger": "LEGAL_REGULATORY_RULE"
                }
                
        # Rule 2: Active Security Compromise / Fraud / Theft
        for pat in self.SECURITY_FRAUD_PATTERNS:
            if pat.search(msg):
                return {
                    "decision": "ESCALATE",
                    "reason": "High-risk account takeover, fraud, or stolen shipment issue.",
                    "confidence": 0.98,
                    "trigger": "SECURITY_FRAUD_RULE"
                }

        # Rule 3: Physical Safety Hazard
        for pat in self.SAFETY_HAZARD_PATTERNS:
            if pat.search(msg):
                return {
                    "decision": "ESCALATE",
                    "reason": "Physical safety or hazardous goods defect requiring immediate specialist handling.",
                    "confidence": 0.99,
                    "trigger": "SAFETY_HAZARD_RULE"
                }

        # Rule 4: Explicit demand for human supervisor or severe service complaint
        for pat in self.HIGH_DISSATISFACTION_PATTERNS:
            if pat.search(msg):
                return {
                    "decision": "ESCALATE",
                    "reason": "Severe service dissatisfaction, repeated unresolved contacts, or explicit human agent demand.",
                    "confidence": 0.95,
                    "trigger": "AGENT_DISSATISFACTION_RULE"
                }

        # Rule 5: Carrier conduct, delivery dispute, or damaged delivery
        for pat in self.CARRIER_DISPUTE_PATTERNS:
            if pat.search(msg):
                return {
                    "decision": "ESCALATE",
                    "reason": "Delivery dispute, driver misconduct, or carrier investigation required.",
                    "confidence": 0.92,
                    "trigger": "CARRIER_DISPUTE_RULE"
                }

        # Rule 6: Billing & Refund Discrepancy
        for pat in self.BILLING_DISPUTE_PATTERNS:
            if pat.search(msg):
                return {
                    "decision": "ESCALATE",
                    "reason": "Disputed refund amount, SLA breach, or duplicate billing issue.",
                    "confidence": 0.93,
                    "trigger": "BILLING_DISPUTE_RULE"
                }

        # Rule 7: Intent-based baseline escalation
        if intent == "service_complaint_escalation":
            return {
                "decision": "ESCALATE",
                "reason": "Inquiry classified under service complaint escalation.",
                "confidence": 0.90,
                "trigger": "INTENT_PRIOR"
            }
            
        if intent == "account_and_payment_security":
            if re.search(r'\b(locked out|2fa|otp.*not coming|cant login|change phone number|closed.*balance|unknown person|scam|scammed)\b', msg, re.IGNORECASE):
                return {
                    "decision": "ESCALATE",
                    "reason": "Customer locked out of authentication or account security anomaly.",
                    "confidence": 0.92,
                    "trigger": "AUTH_LOCKOUT_RULE"
                }

        if intent == "return_and_replacement":
            if re.search(r'\b(third replacement|personal item|someone else\'s|medical|refused my return|a-to-z|seller refused)\b', msg, re.IGNORECASE):
                return {
                    "decision": "ESCALATE",
                    "reason": "Repeated defective items, privacy breach, or marketplace seller dispute.",
                    "confidence": 0.92,
                    "trigger": "RETURN_EXCEPTION_RULE"
                }

        if intent == "prime_and_subscription":
            if re.search(r'\b(charged.*trial|two different accounts|full refund for prime|unknown recurring|charged \$14\.99)\b', msg, re.IGNORECASE):
                return {
                    "decision": "ESCALATE",
                    "reason": "Subscription billing discrepancy or compensation dispute.",
                    "confidence": 0.91,
                    "trigger": "SUBSCRIPTION_DISPUTE_RULE"
                }

        # Rule 8: Classifier Ambiguity / Low Confidence
        if intent_confidence < self.min_intent_conf:
            return {
                "decision": "ESCALATE",
                "reason": f"Intent classification confidence ({intent_confidence:.2f}) below threshold.",
                "confidence": round(1.0 - intent_confidence, 2),
                "trigger": "UNCERTAINTY_FALLBACK"
            }

        # Rule 9: Retrieval Out-of-Distribution Check
        if retrieval_results is not None and len(retrieval_results) > 0:
            top_score = retrieval_results[0].get("score", 0.0)
            if top_score < self.min_retrieval_sim:
                return {
                    "decision": "ESCALATE",
                    "reason": f"Historical retrieval similarity ({top_score:.2f}) insufficient for grounded response.",
                    "confidence": 0.85,
                    "trigger": "OUT_OF_DISTRIBUTION_RETRIEVAL"
                }

        # Default: Safe for automated handling
        return {
            "decision": "AUTO_HANDLE",
            "reason": "Standard operational inquiry with established resolution guidelines and high confidence.",
            "confidence": round(intent_confidence, 2),
            "trigger": "STANDARD_AUTO_HANDLE"
        }
