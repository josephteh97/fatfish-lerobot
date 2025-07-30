#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_so101_leader import SO101LeaderConfig

logger = logging.getLogger(__name__)


class SO101Leader(Teleoperator):
    """
    SO-101 Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101LeaderConfig
    name = "so101_leader"

    def __init__(self, config: SO101LeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        expected_ids = [1]
        # Check if there are other motors on the bus
        succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=True)
        while not succ:
            input(msg)
            succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=True)

        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor ONLY and press enter.")
            succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=False)
            while not succ:
                input(msg)
                succ, msg = self._check_unexpected_motors_on_bus(expected_ids=expected_ids, raise_on_error=False)            
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
            expected_ids.append(self.bus.motors[motor].id)

    def _check_unexpected_motors_on_bus(self, expected_ids: list[int], raise_on_error: bool = True) -> None:
        """
        Check if there are other motors on the bus, if there are other motors, stop the setup process.        
        Raises:
            RuntimeError: If there are other motors on the bus, stop the setup process.
        """
        # Ensure the bus is connected
        if not self.bus.is_connected:
            self.bus.connect(handshake=False)
        
        # Scan all motors at the current baudrate
        current_baudrate = self.bus.get_baudrate()
        self.bus.set_baudrate(current_baudrate)
        
        # Scan all motors on the bus
        found_motors = self.bus.broadcast_ping(raise_on_error=False)
        
        if found_motors is None:
            # If the scan fails, try other baudrates
            for baudrate in self.bus.available_baudrates:
                if baudrate == current_baudrate:
                    continue
                    
                self.bus.set_baudrate(baudrate)
                found_motors = self.bus.broadcast_ping(raise_on_error=False)
                if found_motors is not None:
                    break
        
        # Restore the original baudrate
        self.bus.set_baudrate(current_baudrate)
        
        if found_motors is not None:
            # Check if there are other motors on the bus
            unexpected_motors = [motor_id for motor_id in found_motors.keys() if motor_id not in expected_ids]
            
            if unexpected_motors:
                unexpected_motors_str = ", ".join(map(str, sorted(unexpected_motors)))
                if raise_on_error:
                    raise RuntimeError(
                        f"There are unexpected motors on the bus: {unexpected_motors_str}. "
                        f"Seems this arm has been setup before, not necessary to setup again."
                    )
                else:
                    logger.warning(
                        f"There are unexpected motors on the bus: {unexpected_motors_str}. "
                    )
                    return False, "Please unplug the last motor and press ENTER to try again."
            return True, "OK"
        
        return False, "No motors found on the bus, please connect the arm and press ENTER to try again."

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
