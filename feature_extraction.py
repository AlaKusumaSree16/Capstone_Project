import re

def extract_features(payload):
    payload = str(payload)
    features = []

    # 1. Payload length
    features.append(len(payload))

    # 2. Digit count
    features.append(sum(c.isdigit() for c in payload))

    # 3. Special character count
    features.append(len(re.findall(r"[<>'\";|&=%()]", payload)))

    # 4. SQL keyword presence (count)
    sql_keywords = ["select", "union", "or", "and", "drop", "insert", "--"]
    features.append(sum(1 for k in sql_keywords if k in payload.lower()))

    # 5. XSS indicator presence
    xss_keywords = ["<script", "alert", "onerror", "onload", "<img"]
    features.append(sum(1 for k in xss_keywords if k in payload.lower()))

    # 6. Command injection indicators
    cmd_keywords = [";", "&&", "|", "ls", "cat", "whoami", "ping"]
    features.append(sum(1 for k in cmd_keywords if k in payload.lower()))

   
    return features
