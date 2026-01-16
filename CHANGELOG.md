# CHANGELOG

<!-- version list -->

## v0.2.1 (2026-01-16)

### Bug Fixes

- Activate virtualenv before running semantic-release
  ([#184](https://github.com/pollen-robotics/reachy_mini_conversation_app/pull/184),
  [`2976dd6`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/2976dd6dae65a121ecc0d5af504275dffcf5da43))

- **ci**: Fix semantic-release automation in github actions
  ([`3ae1c95`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/3ae1c950322141a2342213fdbf9a13009802e0dc))

### Chores

- Allow 0.x version bumps
  ([`8f181e1`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/8f181e169ea00c922a7eb841b9b1b660250fe4a0))

- Improve semantic-release workflow
  ([`ef597d7`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/ef597d77d6df417588c455834a00796317a8009e))

- **ci**: Fix semantic-release
  ([`174f419`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/174f4190fc5b5c8c1ab3514288226fe8093453ce))

- **ci**: Fix semantic-release publish auth
  ([`3f13dfc`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/3f13dfc7abacd072391a9825cf2067ae9bafceed))

- **ci**: Run semantic-release version to create releases
  ([`0a040d2`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/0a040d285228627f6ee195bfb2b2876772194959))


## v0.2.0 (2026-01-15)

### Bug Fixes

- **audio channels**: Making the play_loop and receive methods robust to audio inputs/outputs shapes
  ([#132](https://github.com/pollen-robotics/reachy_mini_conversation_app/pull/132),
  [`fdb5f7f`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/fdb5f7f07fac9c395a406e99d8ad21ddcccf3c7d))

- **camera tool**: Removing the temporary file creation which was not compatible with Windows. The
  current solution encodes the frame in the JPG format but stays in the RAM.
  ([#123](https://github.com/pollen-robotics/reachy_mini_conversation_app/pull/123),
  [`e2630f8`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/e2630f81df6e6e2c25adfbc2f6c3eae80c32143b))

- **ci**: Run mypy from .venv to avoid wireless extra install
  ([`1b189a2`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/1b189a21b1c596adfd66e6d54a972d7ff6fe80ed))

- **ci**: Run tests from .venv instead of uv run
  ([`b4d69bc`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/b4d69bcea19a49c7a1c241dd503129a85d820be0))

- **mypy**: Fixing mypy errors
  ([`a6ddf4e`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/a6ddf4ee95cda6aa36643ae72afe8b6f8b789568))

- **samplerate**: Removing hardcoded sample rates values and replacing with actual values from the
  robot, uniformizing processing (always on the receiving side), nuking librosa out
  ([`6e24f8f`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/6e24f8f9372cfb510cad59f188bb4853a84b5415))

### Chores

- Update uv.lock file
  ([`5c54619`](https://github.com/pollen-robotics/reachy_mini_conversation_app/commit/5c546191fe487e384e534c46c8bfa48dede8f656))


## v0.1.0 (2025-10-21)

- Initial Release
