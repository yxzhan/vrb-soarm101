# SOARM101 Mujoco Simulation - TODO

## 1. Fix Mujoco Demo Issues
- [ ] Mujoco viewer sometimes can't be killed (process lingers)
- [ ] Pick and place scene: object positions don't match physical setup
- [ ] Mechanical arm self-collision needs to be resolved

## 2. Test Bambot Calibration
- [ ] After switching URDF, verify calibration works on physical robot
- [ ] Compare joint angles between Mujoco simulation and real hardware

## 3. Fix Bambot Export Build
- [ ] Resolve `bambot/website` build/export failure

## 4. Implement Joint States Broadcast
- [ ] Main arm (非夹爪) publishes `/joint_states` for state monitoring

## 5. Design Three Modes
- [ ] **Real Data Collection**: Record demonstrations from physical robot
- [ ] **Sim Data Collection**: Record demonstrations from Mujoco simulation
- [ ] **Policy Control**: Deploy trained policy to control physical robot

## 6. Implement Camera Feature
- [ ] Integrate camera feed into JupyterLab/Mujoco viewer
- [ ] Support real-time visual feedback during teleoperation

## 7. Data Collection → LeRobot Dataset
Record on the backend via ROS, convert to LeRobot offline. The bambot Record panel only
starts/stops episodes.

```
Record panel ──/record/control──▶ recorder node ──▶ one MCAP bag per episode
             ◀──/record/status───                 └─▶ convert script → LeRobotDataset
```

### Recorder node (Python, runs next to rosbridge)
- [ ] `/record/control` (`std_msgs/String`, JSON): `{"cmd":"start","task":"..."}`, `{"cmd":"stop"}`, `{"cmd":"discard"}`
- [ ] `/record/status` (`std_msgs/String`, JSON), 1 Hz + on change: state, episode index, elapsed, per-topic Hz
- [ ] On start: new MCAP bag `episodes/<index>/` + `meta.json` (task); ignore start while recording (report in status)
- [ ] Subscribe with `raw=True`, write with `rosbag2_py.SequentialWriter` stamped with **receive time**
- [ ] Topics: `/joint_command` (action), `/joint_states` (observation.state), `/*/image_raw/compressed` (observation.images.*)
- [ ] Test with a private `ROS_DOMAIN_ID` and a port other than 9090

### Convert script (bag → LeRobot)
- [ ] Resample to a fixed fps (nearest-neighbour on receive time), build frames, `LeRobotDataset.create / add_frame / save_episode`
- [ ] Run lerobot in its own env (pulls in torch; keep it out of the binder image unless needed)
- [ ] Bags stay as raw data so datasets can be regenerated (other fps, newer LeRobot versions)

### Record panel (bambot)
- [ ] Task text, Start episode / Stop / Discard; show episode index, elapsed, recorder topics + Hz
- [ ] Panel state comes from `/record/status`, not from the button press
- [ ] Keep the in-browser record/replay as the offline (no ROS) feature, separate from collection

### Open questions
- [ ] Direct leader mode doesn't publish `/joint_command` → require ROS mode for collection, or have the follower also publish the commands it applies
- [ ] `/joint_states` is the commanded value, not servo readback → read Present Position (addr 56) periodically and publish that; check serial bandwidth
- [ ] Header stamps are browser `Date.now()` (`bambot/hooks/useROS2.ts` `rosNow()`), skewed per device → fine for recording (receive time is used); for correct `header.stamp` add a clock offset from `/rosapi/get_time` at connect
