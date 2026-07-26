import os
import time
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import GeminiModel

# 1. Initialize Gemini 2.5 Flash as your DeepEval Judge
evaluator_model = GeminiModel(
    model="gemini-3.5-flash",
    temperature=0
)

# 2. Define your metrics
faithfulness = FaithfulnessMetric(
    threshold=0.7, 
    model=evaluator_model,
    include_reason=True
)

relevancy = AnswerRelevancyMetric(
    threshold=0.7, 
    model=evaluator_model,
    include_reason=True
)

# 3. Define 10 Clinical Test Cases based on KKH Nursing Protocols
test_cases = [
    # Case 1: Pediatric Vitals (2-year-old)
    LLMTestCase(
        input="What is the normal heart rate range for a 2-year-old child?",
        actual_output="The normal heart rate range for a child aged 1 to 6 years is 75 to 130 beats per minute.",
        retrieval_context=["Heart Rate for children aged 1-6 years: 75 - 130 bpm. Respiratory Rate: 20 - 30 breaths/min."]
    ),
    # Case 2: Fluid Calculation (Holliday-Segar)
    LLMTestCase(
        input="How much daily maintenance fluid does a 15 kg child need?",
        actual_output="For a 15 kg child, the daily fluid requirement is 1250 mL/day.",
        retrieval_context=["Holliday-Segar Formula: 100 mL/kg for first 10 kg + 50 mL/kg for next 10 kg. Total daily = 1000 + 250 = 1250 mL/day."]
    ),
    # Case 3: Systolic Blood Pressure
    LLMTestCase(
        input="What is the expected minimum systolic blood pressure for a 5-year-old?",
        actual_output="The expected systolic blood pressure for a 5-year-old is greater than 80 mmHg.",
        retrieval_context=["Expected Systolic BP for children 1 to 10 years old: > 70 + (age * 2) mmHg. For 5 years: 70 + 10 = 80 mmHg."]
    ),
    # Case 4: Neonatal Urine Output
    LLMTestCase(
        input="What is the normal expected minimum urine output for a neonate weighing 3 kg?",
        actual_output="For a 3 kg neonate, normal expected urine output is greater than 1.5 mL/hr.",
        retrieval_context=["Normal expected urine output for neonates is > 0.5 mL/kg/hr. For older children, it is > 1.0 mL/kg/hr."]
    ),
    # Case 5: Parkland Burn Fluid Calculation
    LLMTestCase(
        input="How do you calculate 24-hour Parkland burn fluid for a 10 kg child with 20% burn?",
        actual_output="The total 24-hour fluid volume is 600 to 800 mL. Give 300 to 400 mL in the first 8 hours.",
        retrieval_context=["Parkland formula: 3 to 4 mL * weight (kg) * burn percentage. Give half in first 8 hours, and remaining half in next 16 hours."]
    ),
    # Case 6: Pediatric Vitals (Infant)
    LLMTestCase(
        input="What are the normal vital sign ranges for a 4-month-old infant?",
        actual_output="For a 4-month-old infant, normal Heart Rate is 80-160 bpm and Respiratory Rate is 30-60 breaths/min.",
        retrieval_context=["Infant (<0.5 years): Heart Rate 80 - 160 bpm | Respiratory Rate 30 - 60 breaths/min."]
    ),
    # Case 7: Paracetamol Overdose Decontamination
    LLMTestCase(
        input="When should activated charcoal be given for acute paracetamol poisoning?",
        actual_output="Activated charcoal can be given if the patient presents within 1 to 2 hours of acute paracetamol ingestion.",
        retrieval_context=["Single dose activated charcoal (1 g/kg) may be considered if presenting within 1 to 2 hours of acute paracetamol ingestion."]
    ),
    # Case 8: Fluid Calculation (25 kg child)
    LLMTestCase(
        input="Calculate hourly maintenance fluid for a 25 kg child.",
        actual_output="The hourly fluid requirement for a 25 kg child is 65 mL/hr.",
        retrieval_context=["Hourly fluid rate: 4 mL/kg for first 10 kg (40) + 2 mL/kg for next 10 kg (20) + 1 mL/kg for remaining kg (5) = 65 mL/hr."]
    ),
    # Case 9: Pediatric CPR Ratio
    LLMTestCase(
        input="What is the compression-to-ventilation ratio for two-rescuer pediatric CPR?",
        actual_output="For two-rescuer healthcare provider CPR in children, the compression-to-ventilation ratio is 15:2.",
        retrieval_context=["Healthcare provider CPR ratio: 30:2 for single rescuer, 15:2 for two rescuers in infants and children."]
    ),
    # Case 10: Unconscious Patient Escalation
    LLMTestCase(
        input="What should a nurse do immediately if a pediatric patient stops breathing?",
        actual_output="Immediately activate emergency response/code blue, initiate CPR starting with chest compressions, and escalate to the medical team.",
        retrieval_context=["Emergency Protocol: Recognize respiratory arrest, call for assistance/Code Blue, open airway, initiate ventilation and chest compressions."]
    )
]

# 4. Run the evaluation sequentially with a small delay
if __name__ == "__main__":
    print(f"🚀 Starting DeepEval evaluation on {len(test_cases)} clinical test cases...\n")
    
    for index, test_case in enumerate(test_cases, 1):
        print(f"--- Evaluating Test Case #{index} of {len(test_cases)} ---")
        
        # Run evaluation on one case at a time
        evaluate([test_case], metrics=[faithfulness, relevancy])
        
        # 5-second pause keeps API usage smoothly under 10 RPM
        time.sleep(5)
        
    print("\n Evaluation finished successfully for all 10 test cases!")