from pathlib import Path

from config import LEAGUE_CONFIG
from pitch_oracle_core import __version__


ROOT = Path(__file__).resolve().parents[1]
CORE_REF = "e31c5b9703ef05bad486fc348894fedd64655033"
CORE_VERSION = "1.5.0"


def test_consumer_selects_a_registered_non_epl_league():
    assert LEAGUE_CONFIG.key == "belgium"
    assert LEAGUE_CONFIG.key != "epl"
    assert LEAGUE_CONFIG.football_data_div
    assert LEAGUE_CONFIG.espn_slug


def test_core_pin_is_synchronized_everywhere():
    assert __version__ == CORE_VERSION
    pin = f"pitch-oracle-core[consumer] @ git+https://github.com/gmalbert/pitch-oracle-core.git@{CORE_REF}"
    assert pin in (ROOT / "requirements.txt").read_text()
    assert pin in (ROOT / "requirements-ci.txt").read_text()
    workflow = (ROOT / ".github" / "workflows" / "artifact-pipeline.yml").read_text()
    reusable_workflow = "precompute-consumer.yml@"
    workflow_ref = workflow.split(reusable_workflow, 1)[1].split()[0]
    assert workflow_ref == CORE_REF
    assert f"core_ref: {CORE_REF}" in workflow
