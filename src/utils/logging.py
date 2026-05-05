from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_wandb_run: object | None = None


def init_wandb(
    project: str = "tsfm-peft",
    entity: str | None = None,
    config: dict[str, object] | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """wandb 실험 초기화.

    Args:
        project: wandb 프로젝트 이름.
        entity: wandb 팀/사용자.
        config: 실험 설정 딕셔너리.
        name: 실험 이름.
        tags: 실험 태그.
    """
    global _wandb_run
    try:
        wandb_module = importlib.import_module("wandb")
        init_fn = getattr(wandb_module, "init", None)
        if not callable(init_fn):
            logger.warning("wandb.init을 찾을 수 없습니다. 로깅을 건너뜁니다.")
            return

        _wandb_run = init_fn(
            project=project,
            entity=entity,
            config=config,
            name=name,
            tags=tags,
            reinit=True,
        )
        logger.info(f"wandb 초기화: {project}/{name}")
    except ImportError:
        logger.warning("wandb가 설치되지 않았습니다. 로깅을 건너뜁니다.")
    except Exception as e:
        logger.warning(f"wandb 초기화 실패: {e}. 로깅을 건너뜁니다.")


def log_metrics(
    metrics: dict[str, float],
    step: int | None = None,
    prefix: str = "",
) -> None:
    """메트릭을 wandb에 로깅.

    Args:
        metrics: 메트릭 이름→값 딕셔너리.
        step: 글로벌 스텝.
        prefix: 메트릭 이름 접두사 (예: "train/", "val/").
    """
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

    if _wandb_run is not None:
        try:
            log_fn = getattr(_wandb_run, "log", None)
            if callable(log_fn):
                _ = log_fn(metrics, step=step)
            else:
                wandb_module = importlib.import_module("wandb")
                fallback_log_fn = getattr(wandb_module, "log", None)
                if callable(fallback_log_fn):
                    _ = fallback_log_fn(metrics, step=step)
        except Exception as e:
            logger.warning(f"wandb 로깅 실패: {e}")

    # 항상 Python logging으로도 기록
    metrics_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
    logger.info(f"[step={step}] {metrics_str}")


def log_config(config: dict[str, object]) -> None:
    """설정을 wandb에 기록.

    Args:
        config: 설정 딕셔너리.
    """
    if _wandb_run is not None:
        try:
            config_obj = getattr(_wandb_run, "config", None)
            update_fn = getattr(config_obj, "update", None)
            if callable(update_fn):
                _ = update_fn(config)
        except Exception as e:
            logger.warning(f"wandb config 업데이트 실패: {e}")


def finish_wandb() -> None:
    """wandb 실행 종료."""
    global _wandb_run
    if _wandb_run is not None:
        try:
            finish_fn = getattr(_wandb_run, "finish", None)
            if callable(finish_fn):
                _ = finish_fn()
            else:
                wandb_module = importlib.import_module("wandb")
                fallback_finish_fn = getattr(wandb_module, "finish", None)
                if callable(fallback_finish_fn):
                    _ = fallback_finish_fn()
        except Exception as e:
            logger.warning(f"wandb 종료 실패: {e}")
        _wandb_run = None
