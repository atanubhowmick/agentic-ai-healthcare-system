import time

from log.logger import logger


class SystemMonitor:
    """Tracks response time, latency, and throughput across requests."""

    def __init__(self):
        self.request_times = []

    def log_response_time(self, start_time: float) -> float:
        latency = time.time() - start_time
        self.request_times.append(latency)
        return latency

    def calculate_system_health(self, total_requests: int, failed_requests: int) -> dict:
        failure_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
        avg_latency = (
            sum(self.request_times) / len(self.request_times)
            if self.request_times else 0
        )

        return {
            "avg_latency_seconds":    avg_latency,
            "failure_rate_percentage": failure_rate,
            "throughput_req_per_sec": (
                len(self.request_times) / sum(self.request_times)
                if sum(self.request_times) > 0 else 0
            ),
        }
