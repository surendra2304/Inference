class InferenceRuntimeError(Exception):
    pass


class AdmissionRejected(InferenceRuntimeError):
    pass


class ProviderUnavailable(InferenceRuntimeError):
    pass


class BudgetExceeded(InferenceRuntimeError):
    pass


class RateLimitExceeded(InferenceRuntimeError):
    pass


class SchemaViolation(InferenceRuntimeError):
    pass


class CachePolicyError(InferenceRuntimeError):
    pass
