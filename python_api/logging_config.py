"""Logging configuration with JSON output and JSON file handler"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pythonjsonlogger.json import JsonFormatter
import threading


class JSONFileHandler(logging.Handler):
    """Custom logging handler that stores logs in JSON files"""

    def __init__(
        self,
        log_directory: str = "logs",
        level: int = logging.INFO,
        filter_training_only: bool = True,
    ):
        super().__init__(level)
        self.log_directory = log_directory
        self.filter_training_only = filter_training_only
        self._locks = {}
        self._lock = threading.Lock()

        # Create log directory if it doesn't exist
        os.makedirs(self.log_directory, exist_ok=True)

    def get_lock_for_project(self, project_id: str) -> threading.Lock:
        """Get or create a lock for a specific project"""
        with self._lock:
            if project_id not in self._locks:
                self._locks[project_id] = threading.Lock()
            return self._locks[project_id]

    def emit(self, record: logging.LogRecord) -> None:
        """Store log record in JSON file"""
        try:
            # Skip if filtering and no project_id present
            if self.filter_training_only and not hasattr(record, "project_id"):
                return

            project_id = getattr(record, "project_id", "general")

            # Convert log record to dict
            log_entry = self.format_log_entry(record)

            # Write to project-specific JSON file
            log_file = os.path.join(self.log_directory, f"{project_id}.json")

            # Use project-specific lock for thread safety
            lock = self.get_lock_for_project(project_id)
            with lock:
                # Append to JSON file (create if doesn't exist)
                with open(log_file, "a") as f:
                    json.dump(log_entry, f)
                    f.write("\n")

        except Exception:
            # Fall back to stderr if file write fails
            self.handleError(record)

    def format_log_entry(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Format log record for JSON storage"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "logger_name": record.name,
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
        }

        # Add project_id if present
        if hasattr(record, "project_id"):
            log_entry["project_id"] = record.project_id

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "pathname",
                "processName",
                "process",
                "threadName",
                "thread",
                "taskName",
                "project_id",  # Already handled above
            ] and not key.startswith("_"):
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry["exc_info"] = self.format(record)

        return log_entry


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json_file_logging: bool = True,
    json_log_directory: str = "logs",
    json_log_min_level: str = "INFO",
) -> None:
    """Configure logging with JSON output and optional JSON file storage

    Args:
        log_level: Minimum log level for console/file output
        log_file: Optional log file path
        enable_json_file_logging: Whether to enable JSON file logging
        json_log_directory: Directory to store JSON log files
        json_log_min_level: Minimum log level for JSON file storage
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # JSON formatter with custom fields
    json_formatter = JsonFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s",
        rename_fields={
            "timestamp": "@timestamp",
            "level": "level",
            "name": "logger",
        },
        static_fields={"service": "xgboost_training_api"},
    )

    # Console handler with JSON output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)

    # JSON file handler for training logs
    if enable_json_file_logging:
        json_file_handler = JSONFileHandler(
            log_directory=json_log_directory,
            level=getattr(logging, json_log_min_level.upper()),
            filter_training_only=True,
        )
        # Use standard formatter for JSON file (it handles its own formatting)
        json_file_handler.setFormatter(logging.Formatter())
        root_logger.addHandler(json_file_handler)

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def log_with_project_id(
    logger: logging.Logger, project_id: str, message: str, **kwargs
):
    """Helper function to log with project_id in extra fields"""
    extra = kwargs.pop("extra", {})
    extra["project_id"] = project_id
    return logger.info(message, extra=extra, **kwargs)


def read_project_logs(
    project_id: str, log_directory: str = "logs", limit: int = 100
) -> list[Dict[str, Any]]:
    """Read logs for a specific project from JSON file

    Args:
        project_id: The project ID to read logs for
        log_directory: Directory where logs are stored
        limit: Maximum number of logs to return (most recent first)

    Returns:
        List of log entries as dictionaries
    """
    log_file = os.path.join(log_directory, f"{project_id}.json")

    if not os.path.exists(log_file):
        return []

    logs = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log_entry = json.loads(line)
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

        # Sort by timestamp (most recent first) and limit
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return logs[:limit]

    except Exception as e:
        # Return empty list on any error
        print(f"Error reading logs for project {project_id}: {e}")
        return []


# Initialize logging on module import
if os.getenv("DISABLE_AUTO_LOGGING_SETUP") != "true":
    setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE"),
        enable_json_file_logging=os.getenv("ENABLE_JSON_FILE_LOGGING", "true").lower()
        == "true",
        json_log_directory=os.getenv("JSON_LOG_DIRECTORY", "logs"),
        json_log_min_level=os.getenv("JSON_LOG_LEVEL", "INFO"),
    )
