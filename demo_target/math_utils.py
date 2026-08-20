def robust_divide(a: float, b: float) -> float:
    """Divides a by b. Should handle divide by zero gracefully by returning 0.0"""
    if b == 0:
        return 0.0
    return a / b

def sort_positive_numbers(numbers: list[int]) -> list[int]:
    """Returns a sorted list of only the positive numbers from the input."""
    filtered = [n for n in numbers if n > 0]
    filtered.sort()
    return filtered