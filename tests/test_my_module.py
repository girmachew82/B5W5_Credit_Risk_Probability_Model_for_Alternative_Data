# test_my_module.py

def test_addition():
    """This is a simple test function that will be collected by pytest."""
    result = 2 + 2
    assert result == 4

def check_subtraction():
    """This function will NOT be collected by pytest because it doesn't start with 'test_'."""
    result = 5 - 3
    assert result == 2