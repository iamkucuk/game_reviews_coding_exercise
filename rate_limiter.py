import time
import threading
from datetime import datetime, timedelta
from collections import deque

# Rate limiter class for LLM API calls
class RateLimiter:
    def __init__(self, max_calls, period=60):
        """
        Initialize a rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds (default: 60s for per-minute limiting)
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()
    
    def __call__(self, func):
        """
        Decorator for rate limiting.
        """
        def wrapped(*args, **kwargs):
            self.wait_if_needed()
            result = func(*args, **kwargs)
            return result
        return wrapped
    
    def wait_if_needed(self):
        """
        Check if we need to wait to respect rate limits, and if so, wait.
        """
        with self.lock:
            now = datetime.now()
            
            # Remove calls that are outside the time window
            while self.calls and self.calls[0] < now - timedelta(seconds=self.period):
                self.calls.popleft()
            
            # If we've reached the limit, wait until we can make another call
            if len(self.calls) >= self.max_calls:
                wait_time = (self.calls[0] + timedelta(seconds=self.period) - now).total_seconds()
                if wait_time > 0:
                    print(f"Rate limit hit, waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
            
            # Add the current call
            self.calls.append(now)