from nexus.extensions.agents.local_one_agent.error_codes import LocalToolErrorCode


class TestLocalToolErrorCode:
    def test_all_codes_are_strings(self):
        for code in LocalToolErrorCode:
            assert isinstance(code.value, str)

    def test_permission_denied_value(self):
        assert LocalToolErrorCode.PERMISSION_DENIED == 'PERMISSION_DENIED'

    def test_user_rejected_value(self):
        assert LocalToolErrorCode.USER_REJECTED == 'USER_REJECTED'

    def test_no_changes_value(self):
        assert LocalToolErrorCode.NO_CHANGES == 'NO_CHANGES'

    def test_output_too_large_value(self):
        assert LocalToolErrorCode.OUTPUT_TOO_LARGE == 'OUTPUT_TOO_LARGE'

    def test_unsupported_file_type_value(self):
        assert LocalToolErrorCode.UNSUPPORTED_FILE_TYPE == 'UNSUPPORTED_FILE_TYPE'

    def test_unique_values(self):
        values = [code.value for code in LocalToolErrorCode]
        assert len(values) == len(set(values)), 'Duplicate error code values'
