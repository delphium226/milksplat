# Release checklist

Manual checks from spec section 10, done on real hardware before each release.
Ticked items note the date and how they were verified. A failure is fixed in
the owning crate, with a regression test, before the release.

- [x] Audio capture of live music on the default device (2026-11-08, verified with the meter example)
- [x] 60 fps steady at default quality (2026-11-08, dev machine: RTX 3090, 60 Hz display)
- [x] Simulated device-loss recovery (2026-11-08, `--debug-lose-device`, recovered in about 0.5 s)
- [ ] Unplug or switch the default audio device while running
- [ ] Bluetooth headphones as the default device
- [ ] Sleep and resume
- [ ] Minimize and restore
- [ ] Multi-monitor with mixed DPI and refresh rate
- [ ] A 144 Hz display
- [ ] With an exclusive-mode audio app running
- [ ] 4K at full quality on target hardware
- [ ] The zip runs on a clean Windows VM without installing anything
