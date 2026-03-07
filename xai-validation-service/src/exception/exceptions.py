class XAIBaseException(Exception):
    """Base exception for the XAI Validator service."""
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class ValidationLLMException(XAIBaseException):
    """Raised when the LLM-based validation call fails."""
    def __init__(self, message: str):
        super().__init__(error_code="VALIDATION_LLM_ERROR", message=message)


class ValidationParseException(XAIBaseException):
    """Raised when the LLM validation response cannot be parsed."""
    def __init__(self, message: str):
        super().__init__(error_code="VALIDATION_PARSE_ERROR", message=message)
