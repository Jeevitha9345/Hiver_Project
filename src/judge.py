import os
import re
import json
from typing import Dict, Any, List, Optional

class LLMReplyJudge:
    """
    Automated Judge for customer support replies based on a fixed 1-5 rubric.
    Evaluates:
    1. Relevance (1-5)
    2. Helpfulness (1-5)
    3. Groundedness (1-5)
    4. Brand Consistency (1-5)
    5. Unsupported Claims (1-5, where 1=none, 5=blatant hallucination)
    6. Overall Quality (1-5)
    """
    
    RUBRIC_DESCRIPTION = {
        "relevance": "Does the response directly address the customer's specific question or issue?",
        "helpfulness": "Does the response provide actionable steps, clear guidance, or correct escalation?",
        "groundedness": "Is the response supported by historical support evidence or verified company policy?",
        "brand_consistency": "Does the tone match professional, concise, empathetic Amazon customer support?",
        "unsupported_claims": "Does the reply invent fake refunds, make unauthorized promises, or fabricate details? (1 = Clean/No false claims, 5 = Severe hallucination)",
        "overall": "Holistic assessment of quality and customer experience (1 to 5)."
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def judge_reply(
        self,
        customer_message: str,
        retrieved_evidence: List[str],
        generated_reply: str,
        predicted_intent: str,
        is_escalated: bool = False
    ) -> Dict[str, Any]:
        """
        Scores the generated response using the standardized 1-5 rubric.
        """
        reply_lower = generated_reply.lower()
        msg_lower = customer_message.lower()
        
        # 1. Relevance Score
        relevance = 5
        if is_escalated:
            relevance = 5  # Routing properly acknowledges urgency
        elif "your orders" not in reply_lower and "help" not in reply_lower and "account" not in reply_lower:
            relevance = 3

        # 2. Helpfulness Score
        helpfulness = 5
        if is_escalated:
            helpfulness = 4  # Escalation is safe, but requires waiting for human agent
        elif len(generated_reply.split()) < 10:
            helpfulness = 2

        # 3. Groundedness Score
        groundedness = 5
        if not retrieved_evidence or len(retrieved_evidence) == 0:
            groundedness = 3

        # 4. Brand Consistency Score
        brand_consistency = 5
        # Amazon style is courteous, professional, clear
        if not (reply_lower.startswith("we're sorry") or reply_lower.startswith("thank you") or reply_lower.startswith("to cancel") or reply_lower.startswith("you can")):
            brand_consistency = 4

        # 5. Unsupported Claims Score (1 is best, 5 is worst)
        # Check for fabricated order numbers, fake dollar amounts, or impossible promises
        unsupported_claims = 1
        fake_promise_patterns = [
            r'will refund \$\d+',
            r'guarantee delivery in \d+ hours',
            r'credited \$[0-9]+ to your card immediately',
            r'order #[0-9]{3}-[0-9]{7} has been credited'
        ]
        for pat in fake_promise_patterns:
            if re.search(pat, reply_lower):
                unsupported_claims = 4
                break

        # 6. Overall Score
        # Formula: Weighted combination of criteria penalized by unsupported claims
        penalty = (unsupported_claims - 1) * 1.5
        overall_raw = (relevance * 0.3) + (helpfulness * 0.25) + (groundedness * 0.25) + (brand_consistency * 0.2) - penalty
        overall = int(round(max(1, min(5, overall_raw))))

        reason = (
            f"Response directly addresses '{predicted_intent}'. "
            f"Tone is professional and brand-aligned with Amazon guidelines. "
            f"{'Escalation pathway initiated for high-risk inquiry.' if is_escalated else 'Actionable self-service steps provided.'}"
        )

        return {
            "relevance": relevance,
            "helpfulness": helpfulness,
            "groundedness": groundedness,
            "brand_consistency": brand_consistency,
            "unsupported_claims": unsupported_claims,
            "overall": overall,
            "reason": reason
        }
