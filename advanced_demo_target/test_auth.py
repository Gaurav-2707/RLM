import pytest
from auth import login

def test_login_success_8_chars():
    """Test that a password of exactly 8 characters succeeds."""
    result = login(8)
    assert "Session saved" in result

def test_login_success_10_chars():
    """Test that a password of 10 characters succeeds."""
    result = login(10)
    assert "Session saved" in result

def test_login_failure_5_chars():
    """Test that a short password fails."""
    result = login(5)
    assert result == "Login Failed"
