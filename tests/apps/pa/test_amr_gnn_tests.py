"""The AMR-GNN supplement adapter writes the engine's test schema and refuses unmapped isolates."""

from __future__ import annotations

import pandas as pd
import pytest

from bacpredict.apps.pa.amr_gnn_tests import amr_gnn_tests
from bacpredict.engine.ast_labels.ast_tests import TEST_COLUMNS

LONG = pd.DataFrame({
    "run": ["SRR1", "SRR1", "AZPAE2"], "study_id": [None, None, "X"], "study": ["Stanton", "Stanton", "Kos"],
    "ST": ["235", "235", "175"], "drug": ["meropenem", "ciprofloxacin", "Imipenem"],
    "MIC": [">8", "<=0.5", "4"], "RS": ["R", "S", "S"],
})
MAP = pd.DataFrame({"run": ["SRR1", "AZPAE2"], "study": ["Stanton", "Kos"], "ST": ["235", "175"],
                    "sample_accession": ["SAMN1", "SAMEA2"]})


def test_schema_and_values():
    t = amr_gnn_tests(LONG, MAP)
    assert list(t.columns) == TEST_COLUMNS
    assert t.Sample.tolist() == ["SAMN1", "SAMN1", "SAMEA2"]
    assert t.mic_sign.tolist() == [">", "<=", "="] and t.mic_value.tolist() == [8.0, 0.5, 4.0]
    assert t.source_call.tolist() == ["resistant", "susceptible", "susceptible"]
    assert t.drug.tolist()[2] == "imipenem"
    assert t.reads_accession.iloc[0] == "SRR1" and pd.isna(t.reads_accession.iloc[2])  # strain name is not reads


def test_unmapped_isolate_is_an_error():
    with pytest.raises(ValueError, match="no BioSample"):
        amr_gnn_tests(LONG, MAP.iloc[:1])
