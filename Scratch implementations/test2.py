import re


def domain_name(url):
    pattern = r"(?:https?://)?(?:www\.)?([^.]+).(.+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return "invalid format!"


print(domain_name("www.github.com"))
