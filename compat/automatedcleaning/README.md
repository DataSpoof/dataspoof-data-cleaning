# AutomatedCleaning is now MLPilot

This project has a new name: **[MLPilot](https://pypi.org/project/dataspoof-mlpilot/)**. Nothing else
changed.

```bash
pip uninstall AutomatedCleaning && pip install dataspoof-mlpilot
```

```python
import mlpilot as ac        # was: import automatedcleaning as ac
```

This final `AutomatedCleaning` release (2.0.1) contains no code of its own. It installs `dataspoof-mlpilot`
and forwards to it, so the following keep working:
- existing code that runs `import automatedcleaning`;
- the `automatedcleaning` command;
- models saved with AutomatedCleaning.

It receives no further updates. New features and fixes ship in `dataspoof-mlpilot`.

Documentation and source: <https://github.com/DataSpoof/dataspoof-mlpilot>
