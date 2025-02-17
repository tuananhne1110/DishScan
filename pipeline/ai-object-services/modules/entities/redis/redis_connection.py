import json
import logging
from typing import Optional, Dict, Any

import redis
from configs.loader import cfg

# Set up logging
logger = logging.getLogger(__name__)


class RedisConnection:
    """A class to handle Redis connections and operations."""

    def __init__(self, host: str = "localhost", port: int = 6379):
        """Initialize the Redis connection.

        Args:
            host: Redis server host. Defaults to "localhost".
            port: Redis server port. Defaults to 6379.
        """
        self.host = host
        self.port = port
        self.redis_conn = self._initialize_redis()
        self.pre_last_id: Optional[str] = None

    def _initialize_redis(self) -> Optional[redis.Redis]:
        """Initialize and validate the Redis connection.

        Returns:
            A Redis connection object if successful, otherwise None.
        """
        try:
            redis_conn = redis.Redis(host=self.host, port=self.port)
            if redis_conn.ping():
                logger.info("Redis connection initialized successfully.")
                return redis_conn
            else:
                logger.error("Redis server is unavailable.")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}", exc_info=True)
            return None

    def get_last_message(self, topic: str) -> Optional[Dict[str, Any]]:
        """Retrieve the last message from a Redis stream.

        Args:
            topic: The Redis stream topic to read from.

        Returns:
            A dictionary containing the frame and metadata if a new message is found,
            otherwise None.
        """
        if self.redis_conn is None:
            logger.error("Redis connection is not available.")
            return None

        try:
            # Use a pipeline for atomic operations
            pipeline = self.redis_conn.pipeline()
            pipeline.xrevrange(topic, count=1)  # Fix typo: `xreverange` -> `xrevrange`
            messages = pipeline.execute()[0]

            if not messages:
                logger.debug(f"No messages found in topic: {topic}")
                return None

            message_id, message_data = messages[0]

            # Skip if the message is the same as the previous one
            if self.pre_last_id == message_id:
                logger.debug("No new messages found.")
                return None

            self.pre_last_id = message_id

            # Decode the message data
            metadata = json.loads(message_data[b"metadata"])
            frame_bytes = message_data[b"frame"]

            return {
                "frame": frame_bytes,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Failed to retrieve last message from topic {topic}: {e}", exc_info=True)
            return None

    def send_message(self, message: Dict[str, Any], topic: str) -> bool:
        """Send a message to a Redis stream.

        Args:
            message: The message to send, as a dictionary.
            topic: The Redis stream topic to send the message to.

        Returns:
            True if the message was sent successfully, otherwise False.
        """
        if self.redis_conn is None:
            logger.error("Redis connection is not available.")
            return False

        try:
            # Add the message to the stream with a maximum length of 1
            self.redis_conn.xadd(topic, message, maxlen=1)
            logger.debug(f"Message sent successfully to topic: {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to topic {topic}: {e}", exc_info=True)
            return False
