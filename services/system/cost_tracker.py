def calculate_cost(prompt_tokens, completion_tokens):
    # Pricing for gpt-4o-mini (approx)
    input_cost = (prompt_tokens / 1000) * 0.00015
    output_cost = (completion_tokens / 1000) * 0.0006

    total_usd = input_cost + output_cost

    # Convert USD → INR
    usd_to_inr = 83
    total_inr = total_usd * usd_to_inr

    return round(total_inr, 4)