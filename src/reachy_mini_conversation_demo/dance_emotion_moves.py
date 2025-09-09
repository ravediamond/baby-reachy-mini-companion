"""
Dance and emotion moves for the movement queue system.

This module implements dance moves and emotions as Move objects that can be queued
and executed sequentially by the MovementManager.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from reachy_mini_dances_library.dance_move import DanceMove
from reachy_mini.motion.recorded_move import RecordedMoves

from reachy_mini_conversation_demo.moves import Move, FullBodyPose

logger = logging.getLogger(__name__)


class DanceQueueMove(Move):
    """Wrapper for dance moves to work with the movement queue system"""
    
    def __init__(self, move_name: str):
        self.dance_move = DanceMove(move_name)
        super().__init__(self.dance_move.duration)
        self.move_name = move_name
    
    def evaluate(self, t: float) -> FullBodyPose:
        """Evaluate dance move at time t"""
        try:
            # Get the pose from the dance move
            pose_data = self.dance_move.evaluate(t)
            
            # DanceMove returns (head_pose, antennas, body_yaw) - already in correct format
            return pose_data
            
        except Exception as e:
            logger.error(f"Error evaluating dance move '{self.move_name}' at t={t}: {e}")
            # Return neutral pose on error
            from reachy_mini.utils import create_head_pose
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, (0, 0), 0)


class EmotionQueueMove(Move):
    """Wrapper for emotion moves to work with the movement queue system"""
    
    def __init__(self, emotion_name: str, recorded_moves: RecordedMoves):
        self.emotion_move = recorded_moves.get(emotion_name)
        super().__init__(self.emotion_move.duration)
        self.emotion_name = emotion_name
    
    def evaluate(self, t: float) -> FullBodyPose:
        """Evaluate emotion move at time t"""
        try:
            # Get the pose from the emotion move
            pose_data = self.emotion_move.evaluate(t)
            
            # RecordedMove returns (head_pose, antennas, body_yaw) - already in correct format
            return pose_data
            
        except Exception as e:
            logger.error(f"Error evaluating emotion '{self.emotion_name}' at t={t}: {e}")
            # Return neutral pose on error
            from reachy_mini.utils import create_head_pose
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, (0, 0), 0)


class GotoQueueMove(Move):
    """Wrapper for goto moves to work with the movement queue system"""
    
    def __init__(self, target_head_pose: np.ndarray, start_head_pose: np.ndarray = None, 
                 target_antennas: Tuple[float, float] = (0, 0), start_antennas: Tuple[float, float] = None,
                 target_body_yaw: float = 0, start_body_yaw: float = None, duration: float = 1.0):
        super().__init__(duration)
        
        self.target_head_pose = target_head_pose
        self.start_head_pose = start_head_pose
        self.target_antennas = target_antennas
        self.start_antennas = start_antennas or (0, 0)
        self.target_body_yaw = target_body_yaw  
        self.start_body_yaw = start_body_yaw or 0
    
    def evaluate(self, t: float) -> FullBodyPose:
        """Evaluate goto move at time t using linear interpolation"""
        try:
            from reachy_mini.utils.interpolation import linear_pose_interpolation
            from reachy_mini.utils import create_head_pose
            
            # Clamp t to [0, 1] for interpolation
            t_clamped = max(0, min(1, t / self.duration))
            
            # Use start pose if available, otherwise neutral
            if self.start_head_pose is not None:
                start_pose = self.start_head_pose
            else:
                start_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            
            # Interpolate head pose
            head_pose = linear_pose_interpolation(start_pose, self.target_head_pose, t_clamped)
            
            # Interpolate antennas
            antennas = (
                self.start_antennas[0] + (self.target_antennas[0] - self.start_antennas[0]) * t_clamped,
                self.start_antennas[1] + (self.target_antennas[1] - self.start_antennas[1]) * t_clamped
            )
            
            # Interpolate body yaw
            body_yaw = self.start_body_yaw + (self.target_body_yaw - self.start_body_yaw) * t_clamped
            
            return (head_pose, antennas, body_yaw)
            
        except Exception as e:
            logger.error(f"Error evaluating goto move at t={t}: {e}")
            # Return target pose on error
            return (self.target_head_pose, self.target_antennas, self.target_body_yaw)