import re
import numpy as np

from urllib.parse import urlparse
from collections import Counter


def extract_features(url: str) -> dict:
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    path = parsed_url.path
    query = parsed_url.query

    subdomain_parts = hostname.split('.')[:-2]

    domain_keywords = [
        'paypal', 'google', 'apple', 'amazon', 'microsoft', 'facebook', 'instagram', 'linkedin',
        'twitter', 'whatsapp', 'netflix', 'ebay', 'outlook', 'icloud', 'yahoo', 'github',
        'tiktok', 'snapchat', 'discord', 'dropbox', 'skype', 'adobe', 'steam', 'twitch'
    ]

    suspicious_tlds = ['.info', '.top', '.xyz', '.cn', '.ru', '.ml', '.ga', '.gq', '.tk']

    symbols = set('!@#$%^&*()-_=+[]{}|;:,.<>?/\\~')

    return {
        'url_length': len(url),
        'num_of_dots': url.count('.'),
        'sub_domain_lvl': max(len(hostname.split('.')) - 2, 0),
        'path_level': path.count('/'),
        'hostname_length': len(hostname),
        'path_length': len(path),
        'query_length': len(query),

        'num_numeric_chars': sum(c.isdigit() for c in url),
        'num_of_dash_hostname': hostname.count('-'),
        'num_of_symbols': sum(1 for c in url if c in symbols),
        'symbol_ratio': sum(1 for c in url if c in symbols) / len(url) if len(url) > 0 else 0,

        'has_https': int(url.lower().startswith('https')),
        'has_ip_address': int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url))),
        'has_non_ascii': int(any(ord(c) > 127 for c in url)),

        'random_string': int(bool(re.match(r'^[a-zA-Z0-9]{10,}$', path.split('/')[-1]))),

        'domain_in_subdomains': int(any(kw in '.'.join(subdomain_parts).lower() for kw in domain_keywords)),
        'domain_in_paths': int(any(kw in path.lower() for kw in domain_keywords)),
        'brand_in_domain_prefix': int(any(kw in hostname.split('.')[0].lower() for kw in domain_keywords)),

        'login_in_path_or_domain': int('login' in hostname.lower() or 'login' in path.lower()),
        'suspicious_tld': int(any(hostname.endswith(tld) for tld in suspicious_tlds)),

        'avg_token_length': np.mean([len(tok) for tok in re.split(r'\W+', url) if tok]) if any(
            re.split(r'\W+', url)) else 0,
        'entropy': -sum(p * np.log2(p) for p in [freq / len(url) for freq in Counter(url).values()]) if len(
            url) > 0 else 0
    }