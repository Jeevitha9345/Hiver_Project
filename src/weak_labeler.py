import re
import pandas as pd
from typing import Optional, Tuple

class RuleBasedIntentLabeler:
    """
    Domain-specific rule-based intent labeler for AmazonHelp customer interactions.
    Used for weak supervision on the training split to produce training data
    for classical and embedding-based ML classifiers.
    """
    
    RULES = [
        ("service_complaint_escalation", [
            r'\b(worst|pathetic|terrible|horrible|disgusted|unacceptable|disgrace)\b',
            r'\b(supervisor|manager|team lead|human agent|real human)\b',
            r'\b(lawsuit|sue|court|ftc|bbb|better business bureau|legal action)\b',
            r'\b(rude|abrupt|disconnected|hung up|lied|useless bot|robotic)\b',
            r'\b(5th time|4th time|third time|3 weeks|nobody is helping|tired of this)\b'
        ]),
        ("account_and_payment_security", [
            r'\b(hacked|unauthorized|stolen|fraud|phishing|scam|scammed)\b',
            r'\b(password|reset password|forgot password|login|sign in|locked out)\b',
            r'\b(otp|verification code|2fa|two-factor|two step)\b',
            r'\b(gift card|voucher balance|redeem|redeemed)\b',
            r'\b(unknown charge|charged my card|card charged|deducted without)\b'
        ]),
        ("prime_and_subscription", [
            r'\b(prime membership|prime member|cancel prime|renew prime|charged for prime)\b',
            r'\b(prime video|prime music|kindle unlimited|audible|amazon channels)\b',
            r'\b(annual fee|monthly fee|subscription fee|free trial)\b'
        ]),
        ("refund_and_cancellation", [
            r'\b(cancel order|cancellation|cancelled order|cancel my order)\b',
            r'\b(refund|refunded|money back|reimburse|reversal|restocking fee)\b'
        ]),
        ("return_and_replacement", [
            r'\b(broken|shattered|damaged|dented|cracked|leaked|soggy|torn)\b',
            r'\b(return label|return pickup|drop off return|return my item|return this)\b',
            r'\b(wrong item|wrong size|wrong color|different item|missing parts)\b',
            r'\b(replacement|replace|exchange|defective|faulty|won\'t turn on)\b'
        ]),
        ("delivery_tracking_delay", [
            r'\b(tracking|tracking number|carrier|usps|ups|dpd|hermes|courier)\b',
            r'\b(out for delivery|in transit|sorting facility|dispatch|dispatched)\b',
            r'\b(where is my|late delivery|delayed|hasn\'t arrived|not received|still waiting)\b',
            r'\b(delivered to|front door|front porch|neighbor|mailbox|access code)\b'
        ]),
        ("general_product_inquiry", [
            r'\b(in stock|restock|back in stock|out of stock|when will.*available)\b',
            r'\b(warranty|guarantee|specifications|compatible|compatibility)\b',
            r'\b(promo code|coupon|discount|price match|lightning deal)\b',
            r'\b(how do i|can i ship to|international shipping)\b'
        ])
    ]
    
    COMPILED = [(intent, [re.compile(p, re.IGNORECASE) for p in patterns]) for intent, patterns in RULES]

    @classmethod
    def predict_intent(cls, text: str) -> Tuple[str, float]:
        if not isinstance(text, str) or not text.strip():
            return "general_product_inquiry", 0.1
            
        scores = {}
        for intent, regexes in cls.COMPILED:
            matches = sum(1 for r in regexes if r.search(text))
            if matches > 0:
                scores[intent] = matches
                
        if not scores:
            return "general_product_inquiry", 0.3
            
        best_intent = max(scores, key=scores.get)
        confidence = min(0.95, 0.4 + (scores[best_intent] * 0.25))
        return best_intent, round(confidence, 2)
