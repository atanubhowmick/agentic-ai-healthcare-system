class OrchestratorBaseException(Exception):
    """Base exception for the Orchestrator service."""
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class SpecialistUnavailableException(OrchestratorBaseException):
    """Raised when a specialist agent HTTP call fails."""
    def __init__(self, specialist: str, message: str):
        super().__init__(
            error_code="SPECIALIST_UNAVAILABLE",
            message=f"[{specialist}] {message}",
        )


class XAIValidationException(OrchestratorBaseException):
    """Raised when the XAI validator service is unreachable."""
    def __init__(self, message: str):
        super().__init__(error_code="XAI_SERVICE_ERROR", message=message)


class TreatmentServiceException(OrchestratorBaseException):
    """Raised when the treatment agent is unreachable."""
    def __init__(self, message: str):
        super().__init__(error_code="TREATMENT_SERVICE_ERROR", message=message)


class GraphExecutionException(OrchestratorBaseException):
    """Raised when the LangGraph workflow fails unexpectedly."""
    def __init__(self, message: str):
        super().__init__(error_code="GRAPH_EXECUTION_ERROR", message=message)
