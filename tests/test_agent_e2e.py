import pytest
from src.agent import HiverSupportAgent

@pytest.fixture(scope="module")
def agent():
    return HiverSupportAgent()

def test_agent_auto_handle(agent):
    res = agent.process_message("How can I track my recent order status?")
    assert "intent" in res
    assert res["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert "reply" in res
    assert len(res["reply"]) > 20

def test_agent_escalate_fraud(agent):
    res = agent.process_message("My account was hacked and $500 was charged by unknown person!")
    assert res["decision"] == "ESCALATE"
    assert "routing" in res["reply"].lower() or "specialist" in res["reply"].lower()

def test_agent_structure_validity(agent):
    res = agent.process_message("When will the Kindle Paperwhite be back in stock?")
    assert "message" in res
    assert "cleaned_message" in res
    assert "intent" in res
    assert "label" in res["intent"]
    assert "confidence" in res["intent"]
    assert "retrieved_examples" in res
    assert "decision" in res
    assert "decision_reason" in res
    assert "reply" in res
    assert "confidence" in res
