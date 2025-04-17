
from dataclasses import dataclass
import pandas as pd
from typing import Optional, Dict, List
import psutil
import time

@dataclass
class SystemCall:
    call_name: str
    timestamp: float
    duration_ms: float
    process_id: int
    process_name: str
    category: str
    args: str
    error: Optional[str] = None
    return_value: Optional[int] = None

class SystemCallCollector:
    def __init__(self):
        self.calls: List[SystemCall] = []
        
    def start_monitoring(self, selected_calls: List[str] = None):
        """Start monitoring system calls"""
        # This is a simplified version - in production you'd use actual system call tracing
        while True:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    with proc.oneshot():
                        call = SystemCall(
                            call_name='read',  # Example
                            timestamp=time.time(),
                            duration_ms=0.5,
                            process_id=proc.pid,
                            process_name=proc.name(),
                            category='file',
                            args='fd=3, size=1024'
                        )
                        self.calls.append(call)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            time.sleep(0.1)
    
    def stop_monitoring(self):
        """Stop monitoring system calls"""
        return pd.DataFrame([vars(call) for call in self.calls])

    def get_current_data(self) -> pd.DataFrame:
        """Get current monitoring data"""
        return pd.DataFrame([vars(call) for call in self.calls])
