from __future__ import annotations


class IntegrationError(Exception):
    code = "generation_failed"
    http_status = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidRequest(IntegrationError):
    code = "invalid_request"
    http_status = 400


class UnsupportedProtocol(IntegrationError):
    code = "unsupported_protocol"
    http_status = 400


class CheckpointNotFound(IntegrationError):
    code = "checkpoint_not_found"
    http_status = 404


class CheckpointMetadataInvalid(IntegrationError):
    code = "checkpoint_metadata_invalid"
    http_status = 422


class CheckpointIncompatible(IntegrationError):
    code = "checkpoint_incompatible"
    http_status = 422


class CheckpointLoadFailed(IntegrationError):
    code = "checkpoint_load_failed"
    http_status = 500


class GenerationBusy(IntegrationError):
    code = "generation_busy"
    http_status = 409


class GenerationFailed(IntegrationError):
    code = "generation_failed"
    http_status = 500
