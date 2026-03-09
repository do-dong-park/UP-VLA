# =============================================================================
# UP-VLA Calvin 평가 스크립트 (원본 monolithic 버전)
#
# 이 파일은 하나의 프로세스 안에서 모델 로딩 + 환경 구동 + 평가 루프를 모두 처리한다.
# 이후 serve_upvla.py (모델 서버) + eval_upvla.py (환경/평가 클라이언트)로 분해되었다.
# =============================================================================

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# UP-VLA 소스 경로를 sys.path에 추가 (원래 cephfs 경로 — 로컬 실행 시 수정 필요)
sys.path.insert(0, "/cephfs/cjyjk/UnifiedVLM-Manipulation/UP-VLA")
from llava.llava import conversation as conversation_lib
from models import CLIPVisionTower, MAGVITv2, Upvla
from training.prompting_utils import (
    UniversalPrompting_w_action,
    create_attention_mask_predict_next_for_future_prediction,
)
from training.utils import flatten_omega_conf, get_config, image_transform
from transformers import AutoTokenizer, CLIPImageProcessor

# phi1.5 대화 템플릿 설정 (UP-VLA는 phi1.5 기반 LLM을 사용)
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
# SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
#                 "The assistant gives helpful, detailed, and polite answers to the user's questions."
# SYSTEM_PROMPT_LEN = 28
SYSTEM_PROMPT = ""  # 시스템 프롬프트 비활성화 (phi1.5는 빈 시스템 프롬프트 사용)
SYSTEM_PROMPT_LEN = 0

# This is for using the locally installed repo clone when using slurm
# slurm 환경에서 로컬 클론을 우선 참조하도록 경로 추가
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import importlib

import models

importlib.reload(models)  # 로컬 models 패키지를 강제로 재로딩 (경로 충돌 방지)
import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb

# Calvin 환경 관련 유틸리티 (policy_evaluation 패키지)
from policy_evaluation.multistep_sequences import (
    get_sequences,  # 평가용 5-task 시퀀스 생성
)
from policy_evaluation.utils import (
    get_default_beso_and_env,
    get_env_state_for_initial_condition,
    join_vis_lang,
)
from policy_models.rollout.rollout_video import RolloutVideo  # 롤아웃 영상 기록 헬퍼
from policy_models.utils.utils import get_last_checkpoint
from pytorch_lightning import seed_everything
from termcolor import colored
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def get_video_tag(i):
    """분산 평가(multi-GPU) 시 각 워커가 고유한 비디오 태그를 갖도록 인덱스를 조정."""
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def get_log_dir(log_dir):
    """로그 디렉토리를 타임스탬프 하위 폴더로 생성해 반환."""
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results):
    """
    results: 각 시퀀스에서 연속 성공한 subtask 수 (0~5) 리스트

    반환값: [1-chain SR, 2-chain SR, 3-chain SR, 4-chain SR, 5-chain SR]
    i-chain SR = "i개 이상 연속 성공한 시퀀스" / 전체 시퀀스
    """
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        # i 이상 성공한 모든 경우(i, i+1, ..., 5)를 합산
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, cfg, log_dir=None):
    """체크포인트별 평가 결과를 출력하고 JSON으로 저장. wandb에도 로깅."""
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    sequences = get_sequences(cfg.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        epoch = checkpoint
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        # task별 성공/실패 카운트 집계
        cnt_success = Counter()
        cnt_fail = Counter()
        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:  # 성공한 subtask들
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]  # 실패한 첫 subtask
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(
                f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%"
            )

        data = {
            "avg_seq_len": avg_seq_len,
            "chain_sr": chain_sr,
            "task_info": task_info,
        }
        wandb.log(
            {
                "avrg_performance/avg_seq_len": avg_seq_len,
                "avrg_performance/chain_sr": chain_sr,
                "detailed_metrics/task_info": task_info,
            }
        )
        current_data[epoch] = data

        print()

    # 이전 결과와 병합하여 저장 (누적 기록)
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    print(
        f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}"
    )


def evaluate_policy(model, env, lang_embeddings, cfg, num_videos=0, save_dir=None):
    """
    전체 평가 루프.
    num_sequences개의 5-task 시퀀스를 순서대로 돌리고 결과 리스트를 반환.

    model: (model_config, upvla_model, uni_prompting, vq_model, mask_token_id) 튜플
    env: Calvin 환경 객체
    lang_embeddings: 언어 임베딩 조회 객체
    """
    task_oracle = hydra.utils.instantiate(cfg.tasks)  # task 성공 여부 판별기
    val_annotations = cfg.annotations  # task명 → 자연어 어노테이션 dict

    # 비디오 기록 설정
    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    eval_sequences = get_sequences(
        cfg.num_sequences
    )  # (initial_state, eval_sequence) 쌍 목록

    results = []
    plans = defaultdict(list)

    if not cfg.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        record = i < num_videos  # 처음 num_videos개 시퀀스만 영상 기록
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            lang_embeddings,
            val_annotations,
            cfg,
            record,
            rollout_video,
            i,
        )
        results.append(result)
        if record:
            rollout_video.write_to_tmp()
        if not cfg.debug:
            # tqdm 진행바 설명에 실시간 성공률 표시
            success_rates = count_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join(
                [f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)]
            )
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)
        if result < 4 and record:  # 5-task 완전 성공이 아닌 경우 영상 파일로 저장
            rollout_video._log_currentvideos_to_file(i, save_as_video=True)

    return results, plans


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    lang_embeddings,
    val_annotations,
    cfg,
    record,
    rollout_video,
    i,
):
    """
    하나의 5-task 시퀀스를 평가.
    초기 상태로 환경 리셋 후 subtask를 순서대로 rollout.
    첫 실패 시점에서 중단하고 연속 성공 수를 반환.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        if record:
            rollout_video.new_subtask()
        success = rollout(
            env,
            model,
            task_checker,
            cfg,
            subtask,
            lang_embeddings,
            val_annotations,
            record,
            rollout_video,
        )
        if record:
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter  # 실패 → 즉시 종료
    return success_counter


def rollout(
    env,
    model,
    task_oracle,
    cfg,
    subtask,
    lang_embeddings,
    val_annotations,
    record=False,
    rollout_video=None,
):
    """
    단일 subtask 롤아웃 (핵심 함수 — 분해의 기준점).

    act_step마다 모델을 호출해 action_buffer(act_step개 액션)를 채우고,
    매 스텝마다 action_buffer에서 하나씩 꺼내 환경에 적용한다.

    ┌────────────────────────────────────────────────────────┐
    │  serve_upvla.py 로 분리된 부분 (step % act_step == 0)  │
    │  1. 이미지 전처리 ([-1,1] tensor → PIL → VQ 토큰)      │
    │  2. 입력 시퀀스 구성 (text + image tokens → input_ids) │
    │  3. 모델 추론 (pre_pad_predict → gen_token_ids, actions)│
    │  4. 디버그용 예측 이미지 재구성 및 저장                │
    └────────────────────────────────────────────────────────┘
    ┌────────────────────────────────────────────────────────┐
    │  eval_upvla.py 로 분리된 부분 (매 스텝)                 │
    │  5. env.step(action) — 환경 진행                       │
    │  6. task_oracle로 성공 여부 확인                        │
    │  7. 영상 프레임 기록                                    │
    └────────────────────────────────────────────────────────┘
    """
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()
    # subtask에 대응하는 자연어 어노테이션 가져오기 (예: "move the slider to the left")
    lang_annotation = val_annotations[subtask][0]
    # 언어 임베딩 조회 (원본에서는 사용되지만 UP-VLA는 텍스트를 토크나이저로 직접 처리)
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal["lang_text"] = val_annotations[subtask][0]

    # model 튜플 언패킹
    model_config, model, uni_prompting, vq_model, mask_token_id = model

    start_info = env.get_info()  # 태스크 성공 판별을 위한 초기 상태 스냅샷
    batch_size = 1
    action_buffer = None  # [act_step, 7] — act_step개 액션을 한 번에 예측해 버퍼링
    images_to_save_now = None  # 이전 스텝 재구성 이미지 (비교용 PNG 저장에 사용)

    for step in range(cfg.ep_len):
        # ── 모델 추론 블록: act_step 스텝마다 한 번씩 실행 ──────────────────
        if step % model_config.act_step == 0:

            # 1. static 카메라 이미지 전처리
            #    Calvin obs는 [-1, 1] 범위 CHW tensor → HWC numpy → PIL → VQ 토큰
            img_static = (
                obs["rgb_obs"]["rgb_static"].squeeze().permute(1, 2, 0).cpu().numpy()
            )
            image_ori_static = (img_static + 1.0) / 2.0
            image_ori_static *= 255.0
            pixel_values_static = (
                image_transform(
                    Image.fromarray(np.uint8(image_ori_static)),
                    resolution=model_config.dataset.preprocessing.resolution,
                )
                .to(model.device)
                .unsqueeze(0)
            )
            # MAGVITv2로 이미지를 VQ 코드(정수 인덱스)로 양자화
            # text_tokenizer 크기만큼 오프셋을 더해 텍스트 토큰과 공간을 분리
            image_tokens = vq_model.get_code(pixel_values_static) + len(
                uni_prompting.text_tokenizer
            )

            # 2. (선택) gripper 카메라 추가 — num_view=2인 경우
            if model_config.model.vla.num_view == 2:
                img_gripper = (
                    obs["rgb_obs"]["rgb_gripper"]
                    .squeeze()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                image_ori_gripper = (img_gripper + 1.0) / 2.0
                image_ori_gripper *= 255.0
                pixel_values_gripper = (
                    image_transform(
                        Image.fromarray(np.uint8(image_ori_gripper)),
                        resolution=model_config.dataset.preprocessing.resolution,
                    )
                    .to(model.device)
                    .unsqueeze(0)
                )
                image_tokens_gripper = vq_model.get_code(pixel_values_gripper) + len(
                    uni_prompting.text_tokenizer
                )
                # static + gripper 토큰을 dim=1 방향으로 concat
                image_tokens = torch.cat([image_tokens, image_tokens_gripper], dim=1)

            # 3. 입력 시퀀스 구성
            #    uni_prompting: [텍스트 명령 + 이미지 토큰] → input_ids (LLM 입력 형식)
            instruction = goal["lang_text"]
            input_ids, _ = uni_prompting(([instruction], image_tokens), "pre_gen")

            # 4. 어텐션 마스크 생성
            #    이미지 토큰 영역(SOI~EOI)에서 패딩을 제거하는 특수 마스크
            attention_mask = create_attention_mask_predict_next_for_future_prediction(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                rm_pad_in_image=True,
            )

            # 5. 모델 추론
            #    pre_pad_predict: 미래 이미지 토큰 예측 + 액션 예측을 동시에 수행
            #    gen_token_ids: 모델이 예측한 다음 프레임의 VQ 토큰 [1, num_vq_tokens]
            #    actions: 예측된 액션 시퀀스 [1, act_step, 7]
            with torch.no_grad():
                gen_token_ids, actions = model.pre_pad_predict(
                    input_ids=input_ids,
                    uncond_input_ids=None,
                    attention_mask=attention_mask,
                    guidance_scale=None,
                    temperature=None,
                    timesteps=None,
                    noise_schedule=None,
                    noise_type=None,
                    predict_all_tokens=None,
                    seq_len=model_config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=model_config,
                    return_actions=True,
                )
            # action_buffer에 act_step개 액션을 저장 → 이후 스텝에서 하나씩 소비
            action_buffer = actions.squeeze().detach()  # [act_step, 7]

            # ── 예측 이미지 재구성 (디버그/기록용) ──────────────────────────
            # num_view에 따라 image_tokens, gen_token_ids를 뷰별로 분리
            if model_config.model.vla.num_view == 1:
                image_tokens = [image_tokens]
                gen_token_ids = [gen_token_ids]
            elif model_config.model.vla.num_view == 2:
                image_tokens_ori_static, image_tokens_ori_gripper = image_tokens.chunk(
                    2, dim=1
                )
                image_tokens = [image_tokens_ori_static, image_tokens_ori_gripper]
                gen_token_ids_static, gen_token_ids_gripper = gen_token_ids.chunk(
                    2, dim=1
                )
                gen_token_ids = [gen_token_ids_static, gen_token_ids_gripper]
            else:
                raise NotImplementedError(
                    f"Num-view {model_config.model.vla.num_view} not supported"
                )

            images_to_save_new = []
            for i, (image_tokens_i, gen_token_ids_i) in enumerate(
                zip(image_tokens, gen_token_ids)
            ):
                # 예측 토큰을 코드북 범위로 클램프 후 VQ 디코딩
                gen_token_ids_i = torch.clamp(
                    gen_token_ids_i,
                    max=model_config.model.showo.codebook_size - 1,
                    min=0,
                )
                gen_images = vq_model.decode_code(gen_token_ids_i)
                # [-1, 1] → [0, 255] uint8 변환
                gen_images = torch.clamp((gen_images + 1.0) / 2.0, min=0.0, max=1.0)
                gen_images *= 255.0
                gen_images = (
                    gen_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                )

                # 입력 이미지 VQ 재구성 (원본 이미지가 얼마나 잘 표현됐는지 확인용)
                recons_images = vq_model.decode_code(
                    image_tokens_i - len(uni_prompting.text_tokenizer)
                )
                recons_images = torch.clamp(
                    (recons_images + 1.0) / 2.0, min=0.0, max=1.0
                )
                recons_images *= 255.0
                recons_images = (
                    recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                )
                images_to_save_new.append((recons_images, gen_images))

            # PNG 저장: [이전 재구성 | 이전 예측 | 현재 재구성] 3단 비교 이미지
            # step=0일 때는 images_to_save_now가 None이므로 skip
            if images_to_save_now is not None and record:
                images_to_save = [
                    np.concatenate(
                        [recons_images_before, gen_images_before, recons_images_new],
                        axis=2,
                    )
                    for (recons_images_before, gen_images_before), (
                        recons_images_new,
                        _,
                    ) in zip(images_to_save_now, images_to_save_new)
                ]
                images_to_save = np.concatenate(images_to_save, axis=1)
                pil_images = Image.fromarray(images_to_save.squeeze())
                os.makedirs(
                    f"{str(rollout_video.save_dir)}/input_predict_truth", exist_ok=True
                )
                save_path = f"{str(rollout_video.save_dir)}/input_predict_truth/{instruction}_step_{step:03}.png"
                pil_images.save(save_path)

            images_to_save_now = images_to_save_new  # 다음 추론 시 "이전" 이미지로 활용

        # ── 환경 스텝 ────────────────────────────────────────────────────────
        # action_buffer[step % act_step]: 버퍼에서 현재 스텝에 해당하는 액션 선택
        obs, _, _, current_info = env.step(action_buffer[step % model_config.act_step])

        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)

        if record:
            rollout_video.update(obs["rgb_obs"]["rgb_static"])

        # ── 태스크 성공 판별 ─────────────────────────────────────────────────
        # task_oracle: (시작 상태, 현재 상태, 타겟 태스크 집합) → 달성된 태스크 집합
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True  # 성공

    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record:
        rollout_video.add_language_instruction(lang_annotation)
    return False  # ep_len 소진 → 실패


@hydra.main(config_path="../policy_conf", config_name="calvin_evaluate_upvla")
def main(cfg):
    """
    진입점. Hydra로 설정 로딩 → 환경/모델 초기화 → 평가 실행.

    cfg.model_config: upvla_model.yaml 경로 (모델 아키텍처 및 체크포인트 경로 포함)
    cfg.dataset_path: Calvin 데이터셋 경로
    cfg.device: CUDA 디바이스 번호
    """
    log_wandb = cfg.log_wandb
    torch.cuda.set_device(cfg.device)
    seed_everything(0, workers=True)  # type:ignore

    from omegaconf import OmegaConf

    model_config = OmegaConf.load(cfg.model_config)

    lang_embeddings = None
    env = None
    results = {}
    plans = {}
    print(cfg.device)

    # Calvin 환경 + 언어 임베딩 초기화
    env, _, lang_embeddings = get_default_beso_and_env(
        dataset_path=cfg.dataset_path,
        env=env,
        lang_embeddings=lang_embeddings,
        device_id=cfg.device,
        cfg=cfg,
    )

    device = torch.device(f"cuda:{cfg.device}")
    checkpoint = model_config.model.showo.tuned_model_path
    model = get_upvla_agent(model_config, cfg)  # 모델 로딩
    if log_wandb:
        log_dir = get_log_dir(
            model_config.model.showo.tuned_model_path + "/calvin_evaluation"
        )
        os.makedirs(log_dir / "wandb", exist_ok=False)
        results[checkpoint], plans[checkpoint] = evaluate_policy(
            model,
            env,
            lang_embeddings,
            cfg,
            num_videos=cfg.num_videos,
            save_dir=Path(log_dir),
        )


def get_upvla_agent(model_config, cfg):
    """
    UP-VLA 모델 구성요소를 로딩하고 튜플로 반환.

    반환값: (model_config, model, uni_prompting, vq_model, mask_token_id)
      - model_config : OmegaConf 설정 객체
      - model        : Upvla (LLM 기반 VLA 본체)
      - uni_prompting: UniversalPrompting_w_action (텍스트+이미지 토큰 시퀀스 조립기)
      - vq_model     : MAGVITv2 (이미지 ↔ VQ 토큰 변환기)
      - mask_token_id: 마스크 토큰 ID (마스크 예측에 사용)
    """
    device = torch.device(f"cuda:{cfg.device}")
    config = model_config

    # 1. 텍스트 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path, padding_side="left"
    )

    # 2. UniversalPrompting 초기화
    #    텍스트 명령 + 이미지 VQ 토큰을 하나의 입력 시퀀스로 합치는 역할
    #    special_tokens: 이미지 경계(<|soi|>/<|eoi|>) 등 멀티모달 제어 토큰
    #    future_steps=act_step: 액션 예측 스텝 수 (act_step개 액션을 한 번에 예측)
    uni_prompting = UniversalPrompting_w_action(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        future_steps=config.act_step,
    )

    # 3. VQ 모델 (MAGVITv2) 로딩
    #    이미지를 이산 코드(정수 인덱스)로 양자화 / 복원하는 토크나이저
    #    학습하지 않으므로 requires_grad=False
    def get_vq_model_class(model_type):
        if model_type == "magvitv2":
            return MAGVITv2
        else:
            raise ValueError(f"model_type {model_type} not supported.")

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # 4. Upvla 본체 로딩 (사전학습 가중치 기반)
    model = Upvla.from_pretrained(
        config.model.showo.pretrained_model_path,
        low_cpu_mem_usage=False,
        act_step=config.act_step,
    ).to(device)
    assert config.model.showo.vocab_size == model.vocab_size

    # 5. 파인튜닝 체크포인트 적재
    #    Accelerate가 저장한 unwrapped_model/pytorch_model.bin 포맷
    path = f"{config.model.showo.tuned_model_path}/unwrapped_model/pytorch_model.bin"
    print(f"Resuming from checkpoint {path}")
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict  # 메모리 해제
    model.eval()

    mask_token_id = model.config.mask_token_id
    return (model_config, model, uni_prompting, vq_model, mask_token_id)


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
