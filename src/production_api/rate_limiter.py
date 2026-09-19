from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize Limiter using client IP address as key
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
