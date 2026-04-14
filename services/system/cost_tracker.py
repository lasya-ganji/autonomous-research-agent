from config.constants.cost_constants import (
    GPT4O_MINI_INPUT_PRICE,
    GPT4O_MINI_OUTPUT_PRICE,
    USD_TO_INR
)


def calculate_cost(prompt_tokens, completion_tokens):
    input_cost = (prompt_tokens / 1000) * GPT4O_MINI_INPUT_PRICE
    output_cost = (completion_tokens / 1000) * GPT4O_MINI_OUTPUT_PRICE

    total_usd = input_cost + output_cost
    total_inr = total_usd * USD_TO_INR

    return round(total_inr, 4)