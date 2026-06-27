from requests.cookies import RequestsCookieJar


def build_cookie_jar() -> RequestsCookieJar:
    """
    Public repo version intentionally ships without real browser cookies.

    If you need authenticated browsing for local experiments, inject cookies from
    an untracked local file or your own environment bootstrap instead of storing
    them in version control.
    """

    return RequestsCookieJar()


COOKIES = build_cookie_jar()
