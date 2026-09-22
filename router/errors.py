"""Exception types for the Superpowers adaptive routing layer."""


class RouterError(Exception):
    """Base class for all router/policy/classifier errors."""


class RouterConfigError(RouterError):
    """Raised when routing/model configuration is missing, malformed, or inconsistent."""


class ClassifierUnavailableError(RouterError):
    """Raised when the configured classifier (e.g. Laya) cannot be loaded or invoked.

    Callers should catch this and fall back to the deterministic classifier
    (see classifier.get_classifier) rather than letting it propagate -- an
    unavailable classifier must never leave a task unrouted.
    """


class PolicyConflictError(RouterError):
    """Raised when a hard floor (required capability) exceeds a hard ceiling
    (configured max_tier / local_only) for the same task.

    Floors must never be undercut and ceilings must never be exceeded; when a
    single task's floor and ceiling conflict there is no safe automatic
    resolution, so routing fails loudly instead of silently picking a side.
    """
