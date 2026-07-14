"""faro per-cell optogenetic inference server.

Hosts a trained ``Seq2ScalarHistory`` (or a stub policy) and serves per-cell
stimulation exposures to the ``faro`` microscopy control system over HTTP.

Entry points:
    python -m optoerk.serving.app        # run the server
    python -m optoerk.serving.smoke_test # exercise it end-to-end

See ``optoerk/serving/README.md`` for the contract, control law, and deployment.
"""
from optoerk.serving.config import ServerConfig
from optoerk.serving.service import InferenceService

__all__ = ["ServerConfig", "InferenceService"]
