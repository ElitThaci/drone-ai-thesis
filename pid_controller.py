import time

class PIDController:
    def __init__(self, kp, ki, kd, output_min, output_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def compute(self, error):
        now = time.perf_counter()
        if self.last_time is None:
            dt = 0.033  # assume 30fps on first call
        else:
            dt = now - self.last_time
        self.last_time = now

        # proportional
        P = self.kp * error

        # integral with windup clamp
        self.integral += error * dt
        self.integral = max(-100, min(100, self.integral))
        I = self.ki * self.integral

        # derivative
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D = self.kd * derivative
        self.prev_error = error

        output = P + I + D
        return max(self.output_min, min(self.output_max, output))

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
