import os
import sys
import pandas as pd
import psutil
import subprocess
import time
import random
from datetime import datetime

class SystemCallMonitor:
    """Class to monitor system calls on the host system"""
    
    def __init__(self):
        """Initialize the system call monitor"""
        self.supported_calls = {
            'open': {'category': 'file', 'description': 'Opens a file'},
            'read': {'category': 'file', 'description': 'Reads from a file descriptor'},
            'write': {'category': 'file', 'description': 'Writes to a file descriptor'},
            'close': {'category': 'file', 'description': 'Closes a file descriptor'},
            'fork': {'category': 'process', 'description': 'Creates a new process'},
            'exec': {'category': 'process', 'description': 'Executes a program'},
            'socket': {'category': 'network', 'description': 'Creates a socket'},
            'connect': {'category': 'network', 'description': 'Connects to a socket'},
            'accept': {'category': 'network', 'description': 'Accepts a connection on a socket'},
            'stat': {'category': 'file', 'description': 'Gets file status'}
        }
        self.last_capture_time = None
    
    def get_system_calls(self, call_types=None):
        """
        Capture system calls of the specified types.
        
        This is a simplified version that simulates the capture of system calls
        as getting actual system calls requires elevated privileges and tools 
        like strace/dtrace that may not be available in all environments.
        
        Args:
            call_types (list): List of system call types to capture. Use ['all'] for all supported types.
            
        Returns:
            pandas.DataFrame: DataFrame containing system call data
        """
        self.last_capture_time = datetime.now()
        
        # If call_types includes 'all', include all supported call types
        if call_types is None or 'all' in call_types:
            call_types = list(self.supported_calls.keys())
        
        # Create sample data based on actual system activity
        calls_data = []
        processes = psutil.process_iter(['pid', 'name', 'username'])
        
        # Get actual running processes
        process_list = [{'pid': p.info['pid'], 'name': p.info['name'], 'username': p.info['username']} 
                        for p in processes]
        
        # Get file operations
        open_files = []
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                if proc.info['open_files']:
                    for f in proc.info['open_files']:
                        open_files.append({
                            'pid': proc.info['pid'],
                            'process_name': proc.info['name'],
                            'path': f.path
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Get network connections
        connections = []
        for conn in psutil.net_connections(kind='all'):
            try:
                process = psutil.Process(conn.pid) if conn.pid else None
                connections.append({
                    'pid': conn.pid,
                    'process_name': process.name() if process else "Unknown",
                    'local_addr': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                    'remote_addr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                    'status': conn.status
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Generate system call data based on actual system activity
        timestamp = datetime.now().timestamp()
        
        # Generate data for file operations
        if 'open' in call_types and open_files:
            for file_op in open_files[:min(20, len(open_files))]:
                calls_data.append({
                    'timestamp': timestamp - random.random() * 0.5,
                    'call_name': 'open',
                    'category': 'file',
                    'process_id': file_op['pid'],
                    'process_name': file_op['process_name'],
                    'args': file_op['path'],
                    'return_value': random.randint(3, 1000),  # File descriptor
                    'duration_ms': random.uniform(0.01, 5.0),
                    'error': None
                })
        
        if 'read' in call_types and open_files:
            for file_op in open_files[:min(30, len(open_files))]:
                calls_data.append({
                    'timestamp': timestamp - random.random() * 0.5,
                    'call_name': 'read',
                    'category': 'file',
                    'process_id': file_op['pid'],
                    'process_name': file_op['process_name'],
                    'args': f"{random.randint(3, 1000)}, {random.randint(1024, 8192)}",  # fd, count
                    'return_value': random.randint(0, 8192),  # Bytes read
                    'duration_ms': random.uniform(0.05, 10.0),
                    'error': None
                })
        
        if 'write' in call_types and open_files:
            for file_op in open_files[:min(25, len(open_files))]:
                calls_data.append({
                    'timestamp': timestamp - random.random() * 0.5,
                    'call_name': 'write',
                    'category': 'file',
                    'process_id': file_op['pid'],
                    'process_name': file_op['process_name'],
                    'args': f"{random.randint(3, 1000)}, {random.randint(1024, 8192)}",  # fd, count
                    'return_value': random.randint(0, 8192),  # Bytes written
                    'duration_ms': random.uniform(0.05, 15.0),
                    'error': None
                })
        
        # Generate network-related calls
        if 'socket' in call_types and connections:
            for conn in connections[:min(15, len(connections))]:
                calls_data.append({
                    'timestamp': timestamp - random.random() * 0.5,
                    'call_name': 'socket',
                    'category': 'network',
                    'process_id': conn['pid'],
                    'process_name': conn['process_name'],
                    'args': 'AF_INET, SOCK_STREAM, 0',
                    'return_value': random.randint(3, 1000),  # Socket fd
                    'duration_ms': random.uniform(0.01, 2.0),
                    'error': None
                })
        
        if 'connect' in call_types and connections:
            for conn in connections[:min(10, len(connections))]:
                if conn['remote_addr']:
                    calls_data.append({
                        'timestamp': timestamp - random.random() * 0.5,
                        'call_name': 'connect',
                        'category': 'network',
                        'process_id': conn['pid'],
                        'process_name': conn['process_name'],
                        'args': f"{random.randint(3, 1000)}, {conn['remote_addr']}",
                        'return_value': 0,  # Success
                        'duration_ms': random.uniform(1.0, 50.0),
                        'error': None
                    })
        
        # Generate process-related calls
        if 'fork' in call_types and process_list:
            for proc in process_list[:min(5, len(process_list))]:
                calls_data.append({
                    'timestamp': timestamp - random.random() * 2.0,
                    'call_name': 'fork',
                    'category': 'process',
                    'process_id': proc['pid'],
                    'process_name': proc['name'],
                    'args': '',
                    'return_value': random.randint(1000, 10000),  # Child PID
                    'duration_ms': random.uniform(0.5, 20.0),
                    'error': None
                })
        
        # Generate other system calls with randomized data
        for call_type in call_types:
            if call_type not in self.supported_calls:
                continue
                
            # Add some random calls to fill out the dataset
            for _ in range(random.randint(5, 20)):
                proc = random.choice(process_list)
                call_info = self.supported_calls[call_type]
                
                # Random error in about 5% of calls
                error = None
                if random.random() < 0.05:
                    errors = ['EACCES', 'EINVAL', 'ENOENT', 'EBADF', 'EAGAIN']
                    error = random.choice(errors)
                    ret_val = -1
                else:
                    ret_val = random.randint(0, 1000)
                
                calls_data.append({
                    'timestamp': timestamp - random.random() * 1.0,
                    'call_name': call_type,
                    'category': call_info['category'],
                    'process_id': proc['pid'],
                    'process_name': proc['name'],
                    'args': self._generate_args_for_call(call_type),
                    'return_value': ret_val,
                    'duration_ms': random.uniform(0.01, 30.0),
                    'error': error
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(calls_data)
        
        # Sort by timestamp
        if not df.empty:
            df = df.sort_values('timestamp', ascending=False)
        
        return df
    
    def _generate_args_for_call(self, call_name):
        """Generate realistic arguments for different system calls"""
        if call_name == 'open':
            paths = ['/etc/passwd', '/proc/stat', '/var/log/syslog', '/tmp/temp_file',
                     '/usr/lib/python3.8/random.py', '/home/user/documents/file.txt']
            return random.choice(paths) + f", flags={random.choice(['O_RDONLY', 'O_RDWR', 'O_WRONLY|O_CREAT'])}"
        
        elif call_name == 'read' or call_name == 'write':
            fd = random.randint(0, 1000)
            size = random.randint(1, 8192)
            return f"fd={fd}, size={size}"
        
        elif call_name == 'close':
            fd = random.randint(0, 1000)
            return f"fd={fd}"
        
        elif call_name == 'socket':
            families = ['AF_INET', 'AF_UNIX', 'AF_INET6']
            types = ['SOCK_STREAM', 'SOCK_DGRAM', 'SOCK_RAW']
            return f"family={random.choice(families)}, type={random.choice(types)}, protocol=0"
        
        elif call_name == 'connect':
            addresses = ['127.0.0.1:80', '192.168.1.1:443', 'example.com:8080']
            return f"fd={random.randint(3, 1000)}, addr={random.choice(addresses)}"
        
        elif call_name == 'accept':
            return f"fd={random.randint(3, 1000)}, addr=0x{random.randint(0, 0xFFFFFF):x}"
        
        elif call_name == 'stat':
            paths = ['/etc/hosts', '/proc/meminfo', '/var/log/auth.log', '/usr/bin/python']
            return random.choice(paths)
        
        elif call_name == 'exec':
            commands = ['/bin/ls', '/usr/bin/grep', '/usr/bin/find', '/bin/cat', '/usr/bin/python']
            return random.choice(commands) + " " + "-la" if random.random() > 0.5 else ""
        
        else:
            return ""

    def get_cpu_usage(self):
        """Get current CPU usage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self):
        """Get current memory usage"""
        return psutil.virtual_memory().percent
