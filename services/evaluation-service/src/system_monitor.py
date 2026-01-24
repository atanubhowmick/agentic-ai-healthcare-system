import time
import logging

logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    Tracks Response Time, Latency, and Throughput. [cite: 434, 435]
    """
    def __init__(self):
        self.request_times = []

    def log_response_time(self, start_time):
        latency = time.time() - start_time
        self.request_times.append(latency)
        return latency

    def calculate_system_health(self, total_requests, failed_requests):
        # Failure rates: Predictions failed by XAI or Human [cite: 436]
        failure_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
        
        # Latency average
        avg_latency = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        
        return {
            "avg_latency_seconds": avg_latency,
            "failure_rate_percentage": failure_rate,
            "throughput_req_per_sec": len(self.request_times) / sum(self.request_times) if sum(self.request_times) > 0 else 0
        }
        