"""Kedro hooks for ML Engine - Kedro 0.19.5 Compatible."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class LoggingHook:
    """Hook for pipeline logging."""
    
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Any,
        catalog: Any,
    ) -> None:
        """Execute before pipeline runs."""
        logger.info("=" * 80)
        logger.info("🚀 Pipeline Starting")
        logger.info("=" * 80)
    
    def after_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Any,
        catalog: Any,
    ) -> None:
        """Execute after pipeline runs successfully."""
        logger.info("=" * 80)
        logger.info("✅ Pipeline Completed Successfully")
        logger.info("=" * 80)
    
    def on_pipeline_error(
        self,
        error: Exception,
        run_params: Dict[str, Any],
        pipeline: Any,
        catalog: Any,
    ) -> None:
        """Execute on pipeline error."""
        logger.error("=" * 80)
        logger.error(f"❌ Pipeline Failed: {str(error)}")
        logger.error("=" * 80)

HOOKS = (LoggingHook(),)
__all__ = ["HOOKS"]
