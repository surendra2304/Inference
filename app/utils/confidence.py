"""Confidence values that say what was actually measured.

Before this module existed, endpoints returned fixed literals (0.94, 0.95, 0.88) on every
successful call, whatever the input and whether or not any check ran. A reported confidence is
only meaningful as one of these named states:

* ``DEGRADED_CONFIDENCE`` (0.0): no output was produced, so there is nothing to be confident in.
* ``UNVERIFIED_MODEL_CONFIDENCE`` (0.55): a model produced text and no check verified it. This
  is a prior, not a calibrated accuracy. It is used only where no verifier exists.
* Anything higher must come from a named check that actually ran, and must say which check.
"""

DEGRADED_CONFIDENCE: float = 0.0
UNVERIFIED_MODEL_CONFIDENCE: float = 0.55
DETERMINISTIC_RULE_CONFIDENCE: float = 1.0
# A curated, human-reviewed knowledge entry (the grounding store) is not a model output and not
# a computed estimate. This is a fixed editorial prior for that provenance, named so that it is
# never mistaken for a measurement.
CURATED_ENTRY_CONFIDENCE: float = 0.99
