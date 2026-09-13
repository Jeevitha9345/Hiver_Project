import pytest
from src.escalation import EscalationPolicy

def test_escalate_legal_threat():
    policy = EscalationPolicy()
    res = policy.evaluate("I am filing a lawsuit in court against Amazon!", "service_complaint_escalation")
    assert res["decision"] == "ESCALATE"
    assert res["trigger"] == "LEGAL_REGULATORY_RULE"

def test_escalate_account_theft():
    policy = EscalationPolicy()
    res = policy.evaluate("Someone hacked my account and made unauthorized purchases!", "account_and_payment_security")
    assert res["decision"] == "ESCALATE"
    assert res["trigger"] == "SECURITY_FRAUD_RULE"

def test_escalate_severe_agent_complaint():
    policy = EscalationPolicy()
    res = policy.evaluate("Your agent was rude and hung up. Speak to a human manager now!", "service_complaint_escalation")
    assert res["decision"] == "ESCALATE"
    assert res["trigger"] == "AGENT_DISSATISFACTION_RULE"

def test_auto_handle_standard_tracking():
    policy = EscalationPolicy()
    res = policy.evaluate(
        "Where is my package? Tracking number TBA123456 has not updated.",
        "delivery_tracking_delay",
        intent_confidence=0.88,
        retrieval_results=[{"score": 0.75}]
    )
    assert res["decision"] == "AUTO_HANDLE"
