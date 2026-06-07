def calculate_average(scores):
    """
    Calculate the average of a list of numeric scores.

    Parameters:
        scores: a list of numbers (could be empty)

    Returns:
        The average value if the list is non-empty,
        or 0 if the list is empty.
    """
    # If the list is empty or None, avoid a division by zero
    if not scores:
        return 0

    # Accumulate the total of all scores
    total = 0
    for score in scores:
        total += score

    # Divide the total by the number of scores to get the average
    return total / len(scores)


def find_duplicate_names(names):
    """
    Find all duplicate names in a list.

    Parameters:
        names: a list of strings

    Returns:
        A list of names that appeared more than once.
        Each duplicate is appended every time it repeats.
    """
    seen = set()        # Tracks names we've seen at least once
    duplicates = []     # Stores names that appear again

    for name in names:
        if name in seen:
            # This name was already seen before; record as duplicate
            duplicates.append(name)
        else:
            # First time we see this name; add to the seen set
            seen.add(name)

    return duplicates


def process_user_records(records):
    """
    Filter a list of user records, keeping only valid adult records.

    A valid record must:
      - contain 'name' and 'age' keys
      - have 'age' as an int
      - have age >= 18

    Additional behavior:
      - prints 'invalid age' if age is negative
      - prints 'age must be int' if age is not an integer
      - prints 'missing fields' if required keys are missing

    Parameters:
        records: a list of dictionaries representing user data

    Returns:
        A list of valid records that meet the above criteria.
    """
    valid = []

    for record in records:
        # Check that required keys exist
        if 'name' in record and 'age' in record:
            # Check that age is an integer
            if isinstance(record['age'], int):
                # Check if the user is an adult
                if record['age'] >= 18:
                    valid.append(record)
                else:
                    # Underage; if it's also negative, log an error
                    if record['age'] < 0:
                        print('invalid age')
            else:
                # Age is present but not an integer
                print('age must be int')
        else:
            # Missing one or both required fields
            print('missing fields')

    return valid


def risky_division(values, divisor):
    """
    Perform different division-based calculations on each value,
    depending on its size and whether the divisor is zero.

    Parameters:
        values: a list of numeric values
        divisor: the number to divide by (may be zero)

    Returns:
        A list of results where each element is:
          - a computed numeric value if divisor is not zero
          - None if divisor is zero
    """
    results = []

    for value in values:
        if divisor != 0:
            # Different behavior depending on how large value is
            if value > 100:
                # Very large values get doubled after division
                results.append((value / divisor) * 2)
            elif value > 50:
                # Medium values are just divided
                results.append(value / divisor)
            else:
                # Small values add 1 before dividing
                results.append((value + 1) / divisor)
        else:
            # Divisor is zero; avoid crashing and record None
            results.append(None)

    return results