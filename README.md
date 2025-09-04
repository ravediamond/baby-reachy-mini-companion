# Reachy Mini conversation demo

Working repo, we should turn this into a ReachyMini app at some point maybe ?

## Installation

```bash
pip install -e .
```

## Run

```bash
reachy-mini-conversation-demo
```

## Runtime toggles
You can pass flags to control runtime behavior:
- `--sim` - run in simulation mode (no real robot needed).
- `--vision` - enable vision system.
- `--head-tracking` - enable head tracking (ignored if `--sim` is active).
- `--debug` - enable debug logging (default is INFO).