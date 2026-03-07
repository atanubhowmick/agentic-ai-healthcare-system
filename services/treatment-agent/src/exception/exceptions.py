class TreatmentBaseException(Exception):
    """Base exception for the Treatment Agent service."""
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class LLMInvocationException(TreatmentBaseException):
    """Raised when the LLM call fails."""
    def __init__(self, message: str):
        super().__init__(error_code="LLM_INVOCATION_ERROR", message=message)


class LLMResponseParseException(TreatmentBaseException):
    """Raised when the LLM response cannot be parsed into the expected JSON structure."""
    def __init__(self, message: str):
        super().__init__(error_code="LLM_RESPONSE_PARSE_ERROR", message=message)
