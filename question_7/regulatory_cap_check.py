import random

# Regulatory emission cap (e.g. 500 tons of CO2)
REGULATORY_CAP = 500.0

def predict_emissions_ml(input_data):
    """
    Simulates an ML model prediction .
    """
    return random.uniform(450, 550)

def apply_domain_knowledge(prediction):
    """
    Checks whether the predicted emission exceeds the regulatory cap.
    If it does, it flags it as a breach and caps it to the allowed max.
    """
    if prediction > REGULATORY_CAP:
        print(f" Potential Breach! Predicted: {prediction:.2f}, Cap: {REGULATORY_CAP}")
        return {
            "final_value": REGULATORY_CAP,
            "breach_flag": True,
            "original_prediction": prediction
        }
    else:
        return {
            "final_value": prediction,
            "breach_flag": False,
            "original_prediction": prediction
        }

# Test
pred = predict_emissions_ml(None)
result = apply_domain_knowledge(pred)
print(result)
