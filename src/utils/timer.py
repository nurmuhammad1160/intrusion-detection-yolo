"""
Timer utility for alarm cooldown
"""
import time


class Timer:
    """Simple timer for tracking cooldown periods"""
    
    def __init__(self, duration: float):
        """
        Args:
            duration: Timer duration in seconds
        """
        self.duration = duration
        self.start_time = None
        self.is_running = False
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.is_running = True
    
    def stop(self):
        """Stop the timer"""
        self.is_running = False
        self.start_time = None
    
    def reset(self):
        """Reset the timer"""
        self.start()
    
    def is_expired(self) -> bool:
        """Check if timer has expired"""
        if not self.is_running or self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        return elapsed >= self.duration
    
    def get_remaining(self) -> float:
        """Get remaining time in seconds"""
        if not self.is_running or self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        remaining = max(0, self.duration - elapsed)
        return remaining