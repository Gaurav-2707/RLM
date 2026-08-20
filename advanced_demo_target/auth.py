from db import save_user_session


def login(password_length: int) -> str:
    """Return a saved session for passwords that meet the 8-character policy."""
    if password_length > 8:
        security_level = 1
        return save_user_session(user_id=123, security_level=security_level)
    return "Login Failed"
