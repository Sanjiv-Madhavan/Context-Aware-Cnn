class DenoisingError(Exception):
    """Base exception for this project."""


class ConfigValidationError(DenoisingError):
    """Raised when runtime configuration is invalid for current environment."""


class QualityGateError(DenoisingError):
    """Raised when model quality gates are not satisfied."""
