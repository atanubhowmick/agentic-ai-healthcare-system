class EvaluationSvcException(Exception):
    """Base exception for the Evaluation Service."""
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(message)
