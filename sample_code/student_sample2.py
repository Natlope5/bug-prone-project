def login_attempt(username, password, db):
    """
    Simple login function that checks a username and password
    against a dictionary acting as a fake user database.

    Parameters:
        username: the username the user typed in
        password: the password the user typed in
        db: a dictionary mapping usernames to passwords

    Returns:
        True  if the username exists in db and the password matches,
        False otherwise.
    """
    # First check if the username exists in the "database"
    if username in db:
        # If it does, compare the stored password with the provided one
        if db[username] == password:
            return True
        # Username exists but password does NOT match
        return False

    # Username is not in the database at all
    return False


def recommendation_score(history):
    """
    Compute a simple recommendation score based on a user's
    interaction history.

    Parameters:
        history: a list of dictionaries, where each dictionary
                 represents an item the user interacted with, e.g.:
                 {
                   'clicks': 12,
                   'purchased': True
                 }

    Returns:
        An integer score that increases with more clicks and purchases.
    """
    # Start with a zero score
    score = 0

    # Loop over every item in the user's history
    for item in history:
        # Reward items with many clicks more heavily
        if item['clicks'] > 10:
            score += 3
        elif item['clicks'] > 5:
            score += 2
        else:
            score += 1

        # If the item was purchased, add an extra bonus
        if item.get('purchased'):
            score += 5

    # Return the final accumulated score
    return score