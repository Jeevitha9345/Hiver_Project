import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.agent import HiverSupportAgent

def find_real_failures():
    print("Finding real failure examples from the evaluation run...")
    gold_df = pd.read_csv("data/golden_set.csv")
    agent = HiverSupportAgent()
    
    intent_failures = []
    escalation_failures = []
    
    for idx, row in gold_df.iterrows():
        res = agent.process_message(row['message'])
        pred_intent = res['intent']['label']
        pred_action = res['decision']
        
        # Check intent error
        if pred_intent != row['intent']:
            intent_failures.append({
                "id": row['id'],
                "subcat": row['subcategory'],
                "message": row['message'],
                "true_intent": row['intent'],
                "pred_intent": pred_intent,
                "confidence": res['intent']['confidence'],
                "expected_action": row['expected_action'],
                "actual_decision": pred_action,
                "top_retrieval": res['retrieved_examples'][0]['historical_customer_issue'] if res['retrieved_examples'] else None
            })
            
        # Check false auto-handle (missed escalation)
        if row['expected_action'] == "ESCALATE" and pred_action == "AUTO_HANDLE":
            escalation_failures.append({
                "id": row['id'],
                "subcat": row['subcategory'],
                "message": row['message'],
                "true_intent": row['intent'],
                "reason": row['reason'],
                "pred_intent": pred_intent
            })
            
    print(f"\nTotal Intent Misclassifications: {len(intent_failures)} / 200")
    print(f"Total False Auto-Handles: {len(escalation_failures)} / 86")
    
    print("\nSample Intent Misclassifications:")
    for f in intent_failures[:8]:
        print(f"[{f['id']}|{f['subcat']}] Msg: \"{f['message'][:70]}...\"")
        print(f"  True: {f['true_intent']} | Pred: {f['pred_intent']} (conf: {f['confidence']})")
        
    print("\nSample False Auto-Handles:")
    for ef in escalation_failures[:6]:
        print(f"[{ef['id']}] Msg: \"{ef['message'][:70]}...\"")
        print(f"  Expected: ESCALATE ({ef['reason']}) | Intent: {ef['true_intent']}")

if __name__ == "__main__":
    find_real_failures()
