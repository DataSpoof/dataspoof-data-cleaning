"""The library was renamed AutomatedCleaning -> MLPilot in 2.1.0: old names must keep working.
(Artifacts saved by the real AutomatedCleaning 2.0.0 were also checked by hand before release.)"""

import pickle
import warnings

import mlpilot as ac


def test_old_import_name_is_the_same_library():
    import automatedcleaning
    import automatedcleaning.monitoring as old_mon
    from automatedcleaning.automl.automl import AutoML
    assert automatedcleaning is ac
    assert old_mon is ac.monitoring and AutoML is ac.AutoML       # same modules, same classes


def test_pickles_that_name_automatedcleaning_still_load():
    # protocol 0 writes "c<module>\n<name>\n": rewrite the module to what 2.0.0 recorded
    raw = pickle.dumps(ac.SplitResult, protocol=0)
    old = raw.replace(b"cmlpilot.splitting\n", b"cautomatedcleaning.splitting\n")
    assert old != raw
    assert pickle.loads(old) is ac.SplitResult


def test_old_warning_class_name():
    assert ac.AutomatedCleaningWarning is ac.MLPilotWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("x", ac.MLPilotWarning)
    assert issubclass(caught[0].category, ac.AutomatedCleaningWarning)
