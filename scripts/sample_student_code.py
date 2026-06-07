def add_numbers(a, b):
    return a + b

def divide_numbers(a, b):
    if b == 0:
        return None
    return a / b

def find_max(values):
    if not values:
        return None
    max_value = values[0]
    for value in values:
        if value > max_value:
            max_value = value
    return max_value

def process_scores(scores):
    total = 0
    count = 0
    for score in scores:
        if score >= 0:
            total += score
            count += 1
    if count == 0:
        return 0
    return total / count