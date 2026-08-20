import pytest
from math_utils import robust_divide, sort_positive_numbers

def test_robust_divide():
    assert robust_divide(10, 2) == 5.0
    assert robust_divide(5, 0) == 0.0  # Should handle zero division

def test_sort_positive_numbers():
    input_nums = [3, -1, 4, -5, 2, 0]
    result = sort_positive_numbers(input_nums)
    assert result == [2, 3, 4]  # Should filter positive (>0) and sort
