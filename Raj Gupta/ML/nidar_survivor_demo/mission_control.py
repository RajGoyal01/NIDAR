"""Mission rotation on the perception thread; never erase existing databases."""
from .reid import AppearanceMatcher
from .survivors import SurvivorManager, new_mission_path
from .tracking import PersonTracker
from .verification import TemporalVerifier


def rotate_mission(manager, tracker, verifier):
    """Prepare fresh state first; old close must commit before handing it over."""
    matcher = AppearanceMatcher(manager.matcher.config, manager.matcher.encoder, retain_gallery=True)
    new_tracker = PersonTracker(tracker.config)
    new_verifier = TemporalVerifier(verifier.config)
    replacement = SurvivorManager(new_mission_path(), matcher, manager.config)
    try:
        manager.close()
    except Exception:
        replacement.close()
        raise
    return replacement, matcher, new_tracker, new_verifier, manager.snapshot()
