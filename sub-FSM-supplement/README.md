# sub-FSM Supplement (Extended)

Supplemental files for voice interaction sub-FSM integration and validation.

## Included
- Core entry and FSM logic:
  - `main.py`
  - `bridge/ros_voice_bridge.py`
  - `bridge/__init__.py`
  - `brain/schemas.py`
- Sub-FSM tests:
  - `scripts/test_order_subfsm_stability.py`
  - `test/real_paused_ordering_switch_test.py`
- Voice launch and startup integration:
  - `src/asr_tts/launch/speech.launch`
  - `src/asr_tts/launch/cade_voice.launch`
  - `run_task5_all.sh`
  - `task5_person_tracker/run_task5_person_follow_voice.sh`
  - `task5_person_tracker/person_following/run_task5_person_follow_voice.sh`

## Source
Copied from sibling project paths under `/home/nvidia/taskfive`.
