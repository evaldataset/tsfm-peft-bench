"""클러스터 분산 디스패처: 34개 RTX 3060 노드에 TSFM 실험을 SSH로 배포.

각 노드는 한 번에 하나의 작업을 실행하며, 완료 후 결과를 로컬로 rsync.
충돌 후 재개 가능한 상태 파일을 유지하고, 실패한 작업을 최대 2회 재시도.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ─── 설정 ────────────────────────────────────────────────────────
IDLE_NODES_FILE = Path("cluster_logs/idle_nodes.txt")
REMOTE_PROJECT = "~/TSFM"
REMOTE_PYTHON = "~/mlenv/bin/python"
LOCAL_RESULTS = Path("results/expansion")
STATE_FILE = Path("cluster_logs/dispatcher_state.json")

RESULT_SYNC_TARGETS: dict[str, str] = {
    "domain": "results/expansion/domain",
    "rank": "results/expansion/rank",
    "locus": "results/expansion/locus",
    "all_dora": "results/expansion/domain",
    "gradient": "results/gradient_analysis",
}

# 모드별 기본 모델 목록
DEFAULT_MODELS: list[str] = ["chronos", "moment", "moirai"]
DEFAULT_METHODS: dict[str, list[str]] = {
    "domain": ["zero_shot", "head", "lora", "adapter", "full_ft"],
    "rank": ["lora"],
    "locus": ["lora"],
    "all_dora": ["dora"],
    "gradient": ["lora", "adapter"],
}
DEFAULT_DOMAINS: list[str] = ["ett_m1", "finance", "smd"]
DEFAULT_SEEDS: list[int] = [42, 123]

logger = logging.getLogger(__name__)


# ─── 데이터클래스 ─────────────────────────────────────────────────

@dataclass
class Job:
    """단일 실험 작업.

    Args:
        job_id: 고유 식별자 (model_method_domain_seed)
        model: 모델명
        method: PEFT 방법명
        domain: 데이터 도메인명
        seed: 랜덤 시드
        mode: 실행 모드 ("domain", "rank", "locus", "all_dora", "gradient")
        batch_size: 배치 크기
        epochs: 학습 에폭 수
        result_file: 예상 원격 결과 파일 경로 (빈 문자열이면 자동 결정)
        retries: 현재까지 재시도 횟수
        status: 작업 상태 ("pending", "running", "completed", "failed")
    """

    job_id: str
    model: str
    method: str
    domain: str
    seed: int
    mode: str
    batch_size: int = 16
    epochs: int = 10
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    result_file: str = ""
    retries: int = 0
    status: str = "pending"


@dataclass
class Node:
    """클러스터 노드.

    Args:
        name: SSH 별칭 (js-306-000 형식)
        ip: 노드 IP 주소
        busy: 현재 작업 실행 중 여부
        current_job: 현재 실행 중인 작업 (없으면 None)
    """

    name: str
    ip: str
    busy: bool = False
    current_job: Job | None = None


# ─── 유틸리티 ─────────────────────────────────────────────────────

def auto_batch(model: str) -> int:
    """모델별 적절한 배치 크기 반환 (RTX 3060 12 GB 기준).

    Args:
        model: 모델명 ("moment", "chronos", "moirai", "timesfm" 등)

    Returns:
        배치 크기 정수
    """
    return 16 if model == "moment" else 8


def _result_filename(job: Job) -> str:
    """작업에 해당하는 원격 결과 파일명 반환.

    Args:
        job: 대상 작업

    Returns:
        결과 JSON 파일명 (확장자 포함)
    """
    # gradient_probe.py saves as {model}_{method}_{domain}_seed{seed}.json (no prefix)
    return f"{job.model}_{job.method}_{job.domain}_seed{job.seed}.json"


def _remote_result_path(job: Job) -> str:
    """원격 결과 파일의 전체 경로 반환.

    Args:
        job: 대상 작업

    Returns:
        원격 경로 문자열 (~/TSFM/... 형식)
    """
    subdir = RESULT_SYNC_TARGETS.get(job.mode, "results/expansion/domain")
    return f"{REMOTE_PROJECT}/{subdir}/{_result_filename(job)}"


def _local_result_path(job: Job) -> Path:
    """로컬 결과 파일의 Path 반환.

    Args:
        job: 대상 작업

    Returns:
        로컬 결과 파일 Path
    """
    subdir = RESULT_SYNC_TARGETS.get(job.mode, "results/expansion/domain")
    return Path(subdir) / _result_filename(job)


# ─── 노드 로드 ────────────────────────────────────────────────────

def load_nodes(nodes_file: Path) -> list[Node]:
    """idle_nodes.txt에서 노드 목록 파싱.

    Args:
        nodes_file: 노드 파일 경로 (형식: NNN IP)

    Returns:
        Node 객체 목록

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        ValueError: 파일 형식이 잘못되었을 때
    """
    if not nodes_file.exists():
        raise FileNotFoundError(f"노드 파일을 찾을 수 없음: {nodes_file}")

    nodes: list[Node] = []
    for lineno, line in enumerate(nodes_file.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"노드 파일 {lineno}번째 줄 형식 오류 (NNN IP 필요): {line!r}"
            )
        node_id, ip = parts
        name = f"js-306-{node_id}"
        nodes.append(Node(name=name, ip=ip))

    logger.info("노드 %d개 로드 완료", len(nodes))
    return nodes


# ─── 작업 목록 생성 ───────────────────────────────────────────────

def build_job_list(args: argparse.Namespace) -> list[Job]:
    """CLI 인수에서 (model, method, domain, seed) 조합 목록 생성.

    이미 로컬 결과 파일이 존재하는 작업은 건너뜀 (재개 지원).

    Args:
        args: argparse로 파싱된 CLI 인수

    Returns:
        실행할 Job 객체 목록
    """
    models: list[str] = (
        [m.strip() for m in args.models.split(",")]
        if args.models
        else DEFAULT_MODELS
    )
    methods: list[str] = (
        [m.strip() for m in args.methods.split(",")]
        if args.methods
        else DEFAULT_METHODS.get(args.mode, ["lora"])
    )
    domains: list[str] = (
        [d.strip() for d in args.domains.split(",")]
        if args.domains
        else DEFAULT_DOMAINS
    )
    seeds: list[int] = (
        [int(s.strip()) for s in args.seeds.split(",")]
        if args.seeds
        else DEFAULT_SEEDS
    )

    batch_size_override: int | None = getattr(args, "batch_size", None)
    epochs: int = getattr(args, "epochs", 10)
    max_train: int | None = getattr(args, "max_train_samples", None)
    max_eval: int | None = getattr(args, "max_eval_samples", None)

    # gradient 모드: --cells 인수 파싱 (model:method:domain 형식)
    if args.mode == "gradient" and hasattr(args, "cells") and args.cells:
        return _build_gradient_jobs(args, seeds, batch_size_override, epochs)

    jobs: list[Job] = []
    for model in models:
        for method in methods:
            for domain in domains:
                for seed in seeds:
                    job_id = f"{model}_{method}_{domain}_seed{seed}"
                    bs = batch_size_override if batch_size_override else auto_batch(model)
                    job = Job(
                        job_id=job_id,
                        model=model,
                        method=method,
                        domain=domain,
                        seed=seed,
                        mode=args.mode,
                        batch_size=bs,
                        epochs=epochs,
                        max_train_samples=max_train,
                        max_eval_samples=max_eval,
                    )
                    job.result_file = _remote_result_path(job)

                    local = _local_result_path(job)
                    if local.exists():
                        logger.debug("결과 파일 존재, 건너뜀: %s", local)
                        continue
                    jobs.append(job)

    logger.info("작업 %d개 생성 (이미 완료된 작업 제외)", len(jobs))
    return jobs


def _build_gradient_jobs(
    args: argparse.Namespace,
    seeds: list[int],
    batch_size_override: int | None,
    epochs: int,
) -> list[Job]:
    """gradient 모드용 작업 목록 생성 (--cells 인수 파싱).

    Args:
        args: CLI 인수
        seeds: 시드 목록
        batch_size_override: 배치 크기 덮어쓰기 값
        epochs: 에폭 수

    Returns:
        gradient 작업 목록
    """
    cells_str: str = args.cells
    jobs: list[Job] = []
    for cell in cells_str.split(","):
        cell = cell.strip()
        parts = cell.split(":")
        if len(parts) != 3:
            logger.warning("cells 형식 오류 (model:method:domain 필요): %r", cell)
            continue
        model, method, domain = parts
        for seed in seeds:
            job_id = f"grad_{model}_{method}_{domain}_seed{seed}"
            bs = batch_size_override if batch_size_override else auto_batch(model)
            job = Job(
                job_id=job_id,
                model=model,
                method=method,
                domain=domain,
                seed=seed,
                mode="gradient",
                batch_size=bs,
                epochs=epochs,
            )
            job.result_file = _remote_result_path(job)

            local = _local_result_path(job)
            if local.exists():
                logger.debug("gradient 결과 파일 존재, 건너뜀: %s", local)
                continue
            jobs.append(job)

    logger.info("gradient 작업 %d개 생성", len(jobs))
    return jobs


# ─── 원격 명령어 구성 ─────────────────────────────────────────────

def build_remote_command(job: Job) -> str:
    """작업에 맞는 원격 SSH 실행 명령어 반환.

    Args:
        job: 실행할 작업

    Returns:
        원격 노드에서 실행할 bash 명령어 문자열
    """
    if job.mode == "gradient":
        cell_spec = f"{job.model}:{job.method}:{job.domain}"
        return (
            f"cd {REMOTE_PROJECT} && PYTHONPATH=. {REMOTE_PYTHON} "
            f"scripts/gradient_probe.py "
            f"--cells '{cell_spec}' "
            f"--seeds {job.seed} "
            f"--epochs {job.epochs} "
            f"--gpu 0 "
            f"--batch_size {job.batch_size} "
            f"--output_dir results/gradient_analysis"
        )

    # domain / rank / locus / all_dora 모드
    mode_flag = "domain" if job.mode == "all_dora" else job.mode
    extra = ""
    if job.max_train_samples is not None:
        extra += f" --max_train_samples {job.max_train_samples}"
    if job.max_eval_samples is not None:
        extra += f" --max_eval_samples {job.max_eval_samples}"
    return (
        f"cd {REMOTE_PROJECT} && PYTHONPATH=. {REMOTE_PYTHON} "
        f"scripts/run_expansion.py "
        f"--mode {mode_flag} "
        f"--models {job.model} "
        f"--methods {job.method} "
        f"--domains {job.domain} "
        f"--seeds {job.seed} "
        f"--gpu 0 "
        f"--batch_size {job.batch_size} "
        f"--epochs {job.epochs}{extra}"
    )


# ─── SSH / rsync 실행 ────────────────────────────────────────────

def _ssh_run(
    node_name: str,
    remote_cmd: str,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """원격 노드에서 명령어 실행.

    Args:
        node_name: SSH 호스트 별칭 (js-306-NNN)
        remote_cmd: 원격에서 실행할 명령어
        timeout: 초 단위 타임아웃 (None이면 무제한)

    Returns:
        subprocess.CompletedProcess 결과
    """
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
        "-o", "BatchMode=yes",
        node_name,
        remote_cmd,
    ]
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _check_remote_result(node: Node, job: Job) -> bool:
    """원격 노드에 결과 파일이 존재하는지 확인.

    Args:
        node: 확인할 노드
        job: 확인할 작업

    Returns:
        결과 파일이 존재하면 True
    """
    check_cmd = f"test -f {job.result_file} && echo FOUND || echo MISSING"
    try:
        result = _ssh_run(node.name, check_cmd, timeout=60)
        return "FOUND" in result.stdout
    except subprocess.TimeoutExpired:
        logger.warning("[%s] 결과 확인 타임아웃: %s", node.name, job.job_id)
        return False


def sync_results_from_node(node: Node, job: Job) -> bool:
    """원격 노드에서 로컬로 결과 파일을 rsync.

    Args:
        node: 원본 노드
        job: 동기화할 작업

    Returns:
        rsync 성공 여부
    """
    subdir = RESULT_SYNC_TARGETS.get(job.mode, "results/expansion/domain")
    local_dir = Path(subdir)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_glob = f"{node.name}:{REMOTE_PROJECT}/{subdir}/*.json"
    cmd = ["rsync", "-az", remote_glob, str(local_dir) + "/"]

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(
            "[%s] rsync 실패 (code=%d): %s",
            node.name,
            result.returncode,
            result.stderr.strip(),
        )
        return False

    logger.info("[%s] rsync 완료: %s -> %s", node.name, job.job_id, subdir)
    return True


# ─── 작업 실행 (블로킹) ───────────────────────────────────────────

def run_job_on_node(node: Node, job: Job) -> bool:
    """노드에서 작업을 실행하고 완료 여부 반환 (블로킹).

    Args:
        node: 실행할 노드
        job: 실행할 작업

    Returns:
        성공 여부 (결과 파일 생성 확인 기준)
    """
    remote_cmd = build_remote_command(job)
    logger.info("[%s] 작업 시작: %s", node.name, job.job_id)
    logger.debug("[%s] 명령어: %s", node.name, remote_cmd)

    try:
        result = _ssh_run(node.name, remote_cmd, timeout=None)
    except subprocess.TimeoutExpired:
        logger.error("[%s] 작업 타임아웃: %s", node.name, job.job_id)
        return False
    except Exception as exc:
        logger.error("[%s] 작업 실행 예외 (%s): %s", node.name, job.job_id, exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "[%s] 비정상 종료 (code=%d, job=%s): %s",
            node.name,
            result.returncode,
            job.job_id,
            result.stderr.strip()[-500:],
        )
        # 결과 파일이 있으면 성공으로 처리 (일부 스크립트가 0이 아닌 코드 반환 가능)
        return _check_remote_result(node, job)

    return _check_remote_result(node, job)


def run_and_sync(node: Node, job: Job) -> tuple[Node, Job, bool]:
    """작업 실행 후 결과 동기화까지 수행.

    Args:
        node: 실행 노드
        job: 실행 작업

    Returns:
        (node, job, success) 튜플
    """
    success = run_job_on_node(node, job)
    if success:
        sync_results_from_node(node, job)
    return node, job, success


# ─── 상태 파일 ───────────────────────────────────────────────────

def save_state(
    pending: list[Job],
    completed: list[Job],
    failed: list[Job],
) -> None:
    """현재 디스패처 상태를 JSON 파일에 저장.

    Args:
        pending: 대기 중인 작업 목록
        completed: 완료된 작업 목록
        failed: 실패한 작업 목록
    """
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "timestamp": time.time(),
        "pending": [asdict(j) for j in pending],
        "completed": [asdict(j) for j in completed],
        "failed": [asdict(j) for j in failed],
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    logger.debug("상태 저장: pending=%d, completed=%d, failed=%d",
                 len(pending), len(completed), len(failed))


def load_state() -> tuple[list[Job], list[Job], list[Job]] | None:
    """저장된 디스패처 상태 파일 로드.

    Returns:
        (pending, completed, failed) 튜플, 파일 없으면 None
    """
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
        pending = [Job(**j) for j in data.get("pending", [])]
        completed = [Job(**j) for j in data.get("completed", [])]
        failed = [Job(**j) for j in data.get("failed", [])]
        logger.info(
            "상태 파일 로드: pending=%d, completed=%d, failed=%d",
            len(pending), len(completed), len(failed),
        )
        return pending, completed, failed
    except Exception as exc:
        logger.warning("상태 파일 로드 실패: %s", exc)
        return None


# ─── 진행 상황 출력 ───────────────────────────────────────────────

def print_progress(
    pending: list[Job],
    running_count: int,
    completed: list[Job],
    failed: list[Job],
    nodes: list[Node],
) -> None:
    """현재 큐 상태를 로거로 출력.

    Args:
        pending: 대기 작업 목록
        running_count: 현재 실행 중인 작업 수
        completed: 완료 작업 목록
        failed: 실패 작업 목록
        nodes: 전체 노드 목록
    """
    busy_nodes = [n.name for n in nodes if n.busy]
    total = len(pending) + running_count + len(completed) + len(failed)
    logger.info(
        "진행: 대기=%d | 실행중=%d | 완료=%d | 실패=%d | 전체=%d | "
        "바쁜노드=%s",
        len(pending),
        running_count,
        len(completed),
        len(failed),
        total,
        ",".join(busy_nodes) if busy_nodes else "없음",
    )


# ─── 메인 디스패처 루프 ───────────────────────────────────────────

def dispatcher_loop(
    nodes: list[Node],
    jobs: list[Job],
    max_retries: int = 2,
    progress_interval: int = 30,
) -> tuple[list[Job], list[Job]]:
    """모든 작업을 노드에 분산 실행하는 메인 루프.

    각 노드는 한 번에 하나의 작업만 처리.
    max_retries 초과 시 해당 작업을 실패로 분류.

    Args:
        nodes: 사용 가능한 노드 목록
        jobs: 실행할 작업 목록
        max_retries: 작업당 최대 재시도 횟수
        progress_interval: 진행 상황 출력 주기 (초)

    Returns:
        (completed, failed) 작업 목록 튜플
    """
    pending: list[Job] = list(jobs)
    completed: list[Job] = []
    failed: list[Job] = []

    # future -> (node, job) 매핑
    future_map: dict[Future[tuple[Node, Job, bool]], tuple[Node, Job]] = {}

    last_progress_time = time.time()
    last_state_save_time = time.time()

    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        while pending or future_map:
            # 유휴 노드에 작업 배정
            for node in nodes:
                if not node.busy and pending:
                    job = pending.pop(0)
                    job.status = "running"
                    node.busy = True
                    node.current_job = job
                    future = executor.submit(run_and_sync, node, job)
                    future_map[future] = (node, job)
                    logger.info(
                        "[%s] 작업 배정: %s (남은 대기: %d)",
                        node.name,
                        job.job_id,
                        len(pending),
                    )

            # 완료된 future 처리 (non-blocking)
            done_futures = [f for f in future_map if f.done()]
            for future in done_futures:
                node, job = future_map.pop(future)
                node.busy = False
                node.current_job = None

                try:
                    _, _, success = future.result()
                except Exception as exc:
                    logger.error("future 예외 (%s): %s", job.job_id, exc)
                    success = False

                if success:
                    job.status = "completed"
                    completed.append(job)
                    logger.info("완료: %s", job.job_id)
                else:
                    job.retries += 1
                    if job.retries <= max_retries:
                        job.status = "pending"
                        pending.append(job)
                        logger.warning(
                            "재시도 예약 (%d/%d): %s",
                            job.retries,
                            max_retries,
                            job.job_id,
                        )
                    else:
                        job.status = "failed"
                        failed.append(job)
                        logger.error(
                            "최대 재시도 초과, 실패 처리: %s",
                            job.job_id,
                        )

            # 진행 상황 주기적 출력
            now = time.time()
            if now - last_progress_time >= progress_interval:
                print_progress(
                    pending,
                    len(future_map),
                    completed,
                    failed,
                    nodes,
                )
                last_progress_time = now

            # 상태 파일 주기적 저장 (60초마다)
            if now - last_state_save_time >= 60:
                save_state(pending, completed, failed)
                last_state_save_time = now

            time.sleep(5)

    # 최종 상태 저장
    save_state([], completed, failed)

    logger.info(
        "디스패치 완료: 성공=%d, 실패=%d",
        len(completed),
        len(failed),
    )
    if failed:
        logger.warning(
            "실패한 작업 목록: %s",
            ", ".join(j.job_id for j in failed),
        )

    return completed, failed


# ─── CLI 파싱 ────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인수 파서 생성.

    Returns:
        구성된 ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="TSFM 클러스터 분산 디스패처 (34× RTX 3060)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # DoRA domain 모드 전체 실행 (60개 작업, 34개 노드)
  python scripts/cluster_dispatch.py --mode domain --models chronos,moment,moirai \\
    --methods dora --domains ett_m1,finance,smd,physionet \\
    --seeds 42,123,7,2024,3407 --batch_size 8

  # gradient probe (12개 작업)
  python scripts/cluster_dispatch.py --mode gradient \\
    --cells "chronos:lora:ett_m1,chronos:adapter:finance,moirai:lora:ett_m1" \\
    --seeds 42,123 --batch_size 8

  # dry-run (작업 목록만 출력)
  python scripts/cluster_dispatch.py --mode domain --methods dora \\
    --domains ett_m1 --seeds 42,123 --dry_run
""",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["domain", "rank", "locus", "gradient", "all_dora"],
        help="실행 모드",
    )
    parser.add_argument(
        "--models",
        default="",
        help="쉼표 구분 모델 목록 (기본값: chronos,moment,moirai)",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="쉼표 구분 PEFT 방법 목록 (기본값: 모드별 자동 결정)",
    )
    parser.add_argument(
        "--domains",
        default="",
        help="쉼표 구분 도메인 목록 (기본값: ett_m1,finance,smd)",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="쉼표 구분 시드 목록 (기본값: 42,123)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="배치 크기 (0이면 모델별 자동 결정: moment=16, 나머지=8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="학습 에폭 수 (기본값: 10)",
    )
    parser.add_argument(
        "--cells",
        default="",
        help="gradient 모드용 셀 지정 (형식: model:method:domain, 쉼표 구분)",
    )
    parser.add_argument(
        "--nodes_file",
        default=str(IDLE_NODES_FILE),
        help=f"노드 목록 파일 (기본값: {IDLE_NODES_FILE})",
    )
    parser.add_argument(
        "--state_file",
        default=str(STATE_FILE),
        help=f"상태 파일 경로 (다중 dispatcher 동시 실행 시 별도 지정, 기본값: {STATE_FILE})",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="SMD/PSM train 서브샘플 상한 (도메인 로더에 전달)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="SMD/PSM val/test 서브샘플 상한 (도메인 로더에 전달)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="작업당 최대 재시도 횟수 (기본값: 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="상태 파일에서 이전 실행 재개",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="작업 목록만 출력하고 실제 실행하지 않음",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본값: INFO)",
    )
    return parser


# ─── 진입점 ─────────────────────────────────────────────────────

def main() -> None:
    """클러스터 디스패처 메인 진입점."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # 배치 크기 0 → None으로 변환 (auto_batch 사용)
    if args.batch_size == 0:
        args.batch_size = None  # type: ignore[assignment]

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    nodes_file = Path(args.nodes_file)
    nodes = load_nodes(nodes_file)

    # state_file CLI 옵션을 전역 변수에 반영 (다중 dispatcher 지원)
    global STATE_FILE
    STATE_FILE = Path(args.state_file)

    # 작업 목록 결정
    if args.resume:
        state = load_state()
        if state is not None:
            pending, completed, failed = state
            logger.info("이전 상태에서 재개: pending=%d", len(pending))
        else:
            logger.info("재개할 상태 파일 없음, 새로 시작")
            pending = build_job_list(args)
            completed = []
            failed = []
    else:
        pending = build_job_list(args)
        completed = []
        failed = []

    if not pending:
        logger.info("실행할 작업이 없습니다 (모두 완료되었거나 비어있음).")
        return

    # dry-run: 작업 목록 출력 후 종료
    if args.dry_run:
        logger.info("=== DRY RUN: 실제 실행하지 않음 ===")
        for job in pending:
            cmd = build_remote_command(job)
            logger.info(
                "  [%s] 예상 노드 없음 | 명령어: %s",
                job.job_id,
                cmd,
            )
        logger.info("총 %d개 작업 대기 중 (노드 %d개 사용 가능)", len(pending), len(nodes))
        return

    logger.info(
        "디스패치 시작: 작업=%d, 노드=%d",
        len(pending),
        len(nodes),
    )

    completed_jobs, failed_jobs = dispatcher_loop(
        nodes=nodes,
        jobs=pending,
        max_retries=args.max_retries,
    )

    # 최종 요약
    logger.info("─── 최종 요약 ───")
    logger.info("  성공: %d개", len(completed_jobs))
    logger.info("  실패: %d개", len(failed_jobs))
    if failed_jobs:
        for job in failed_jobs:
            logger.info("    실패 작업: %s (재시도 %d회)", job.job_id, job.retries)


if __name__ == "__main__":
    main()
