from nexus.runtime.executors.terminal.keys import keys_to_bytes


def test_named_keys():
    assert keys_to_bytes('Enter') == b'\r'
    assert keys_to_bytes('Tab') == b'\t'
    assert keys_to_bytes('Up') == b'\x1b[A'


def test_control_key():
    assert keys_to_bytes('C-c') == b'\x03'
    assert keys_to_bytes('C-a') == b'\x01'


def test_literal_string():
    assert keys_to_bytes('hello') == b'hello'


def test_list_concatenates():
    assert keys_to_bytes(['echo hi', 'Enter']) == b'echo hi\r'
