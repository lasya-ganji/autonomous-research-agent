def retry(func, retries=3):
    for _ in range(retries):
        try:
            return func()
        except Exception:
            continue
    return None