# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


# TODO - SHOULD BE FURTHER IMPROVED.
class UniversalPrompting_w_action:
    """
    텍스트 명령 + 이미지 VQ 토큰을 LLM 입력 시퀀스로 조립하는 클래스.

    평가 시 사용 경로:
      uni_prompting(([instruction], image_tokens), "pre_gen")
        → pre_gen_prompt() 호출
        → input_ids 반환 (어텐션 마스크는 별도 함수로 생성)

    조립되는 시퀀스 구조 (pre_gen):
      [PAD...] [<|t2i|>] [SOT] [텍스트 토큰들] [EOT]  ← max_text_len으로 좌패딩 정렬
      [<|soi|>] [이미지 VQ 토큰 256개] [<|eoi|>]       ← 이미지 영역 (양방향 어텐션)
      [<|lvg|>] × future_steps                          ← 액션 예측 플레이스홀더
    """

    def __init__(
        self,
        text_tokenizer,
        special_tokens=(
            "<|soi|>",  # Start Of Image
            "<|eoi|>",  # End Of Image
            "<|sov|>",  # Start Of Video
            "<|eov|>",  # End Of Video
            "<|t2i|>",  # 태스크 토큰: text-to-image (평가 시 사용)
            "<|mmu|>",  # 태스크 토큰: multimodal understanding
            "<|t2v|>",  # 태스크 토큰: text-to-video
            "<|v2v|>",  # 태스크 토큰: video-to-video
            "<|lvg|>",  # 액션 예측 플레이스홀더 (future_steps개 반복)
        ),
        max_text_len=8000,
        max_seq_len=377,
        ignore_id=-100,
        cond_dropout_prob=0.1,
        future_steps=10,  # act_step과 동일값 — 예측할 액션 수
    ):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        # [PAD] 추가 후 special_tokens 등록 → 토크나이저 어휘 확장
        self.text_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.text_tokenizer.add_tokens(list(special_tokens))
        # sptids_dict: 특수 토큰 문자열 → 정수 ID 매핑 (어텐션 마스크 생성 등에 사용)
        self.sptids_dict = {
            token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token]))
            for token in special_tokens
        }
        self.sptids_dict["<|sot|>"] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict["<|eot|>"] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict["<|pad|>"] = torch.tensor([self.text_tokenizer.pad_token_id])
        # plus 1 because at this time we add a task token before
        # max_text_len+1: 텍스트 앞에 태스크 토큰(<|t2i|>) 1개가 추가되므로 1 더함
        self.max_text_len = max_text_len + 1
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids("[PAD]")
        self.ignore_id = ignore_id
        self.cond_dropout_prob = cond_dropout_prob
        self.future_steps = future_steps

    def pre_prompt(self, text_ids, image_ids, labels):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = (
                [int(self.sptids_dict["<|t2i|>"])]
                + text_ids[i]
                + [self.text_tokenizer.eos_token_id]
            )

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [
                    int(self.sptids_dict["<|t2i|>"]),
                    self.text_tokenizer.bos_token_id,
                    self.text_tokenizer.eos_token_id,
                ]

            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (
                    self.max_text_len - len(temp_ids)
                ) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (
                    len(temp_ids) + image_ids.shape[-1] + 3
                )
            else:
                # should add the eos token
                temp_ids = temp_ids[: self.max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
                temp_masks = [1] * (
                    len(temp_ids) + image_ids.shape[-1] + 3
                )  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat(
                [
                    # should we predict text tokens when doing image reconstruction?
                    torch.tensor(temp_ids).to(device),  # 577
                    self.sptids_dict["<|soi|>"].to(device),
                    labels[i],  # 1024
                    self.sptids_dict["<|eoi|>"].to(device),
                    self.sptids_dict["<|lvg|>"].repeat(self.future_steps).to(device),
                ],
                dim=0,
            )

            temp_label_ids = torch.where(
                temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids
            )
            # for i in range(60):
            #     print(temp_ids[10 * i:])
            temp_ids = torch.cat(
                [
                    torch.tensor(temp_ids).to(device),
                    self.sptids_dict["<|soi|>"].to(device),
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"].to(device),
                    self.sptids_dict["<|lvg|>"].repeat(self.future_steps).to(device),
                ],
                dim=0,
            )

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(attention_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def pre_gen_prompt(self, text_ids, image_ids):
        """
        평가(추론) 시 호출되는 함수 — 레이블 없이 input_ids만 생성.

        조립 순서:
          1. 텍스트 앞에 태스크 토큰 <|t2i|> 추가, 끝에 EOT 추가
          2. max_text_len 맞게 왼쪽에 PAD 채우기 (짧으면 좌패딩, 길면 자름)
          3. [PAD...][<|t2i|>][SOT][텍스트][EOT] + [<|soi|>][이미지][<|eoi|>] + [<|lvg|>×future_steps]
             텍스트 영역의 어텐션 마스크는 PAD=0, 나머지=1
        """
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # temp_ids: [<|t2i|>] [SOT] [텍스트 토큰들] [EOT]
            temp_ids = (
                [int(self.sptids_dict["<|t2i|>"])]
                + text_ids[i]
                + [self.text_tokenizer.eos_token_id]
            )
            if self.max_text_len >= len(temp_ids):
                # 텍스트가 max_text_len보다 짧으면 왼쪽에 PAD 채우기
                temp_ids = [self.pad_id] * (
                    self.max_text_len - len(temp_ids)
                ) + temp_ids
                # 어텐션 마스크: PAD=0, 실제 토큰=1
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * len(
                    temp_ids
                )
            else:
                # 텍스트가 너무 길면 max_text_len-1 위치에서 자르고 EOT 추가
                temp_ids = temp_ids[: self.max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # 최종 시퀀스: [텍스트 영역] [<|soi|>] [이미지 VQ 토큰] [<|eoi|>] [<|lvg|>×future_steps]
            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat(
                [
                    torch.tensor(temp_ids).to(device),
                    self.sptids_dict["<|soi|>"].to(device),
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"].to(device),
                    self.sptids_dict["<|lvg|>"].repeat(self.future_steps).to(device),
                ],
                dim=0,
            )

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    # language modeling
    def lm_prompt(self, text_ids, max_seq_len):

        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                temp_labels_ids = temp_ids + [self.ignore_id] * (
                    max_seq_len - len(temp_ids)
                )
                temp_ids = temp_ids + [self.pad_id] * (max_seq_len - len(temp_ids))
                temp_masks = [1] * len(temp_ids) + [0] * (max_seq_len - len(temp_ids))
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_masks = torch.tensor(temp_masks)
            temp_labels_ids = torch.tensor(temp_labels_ids)

            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(attention_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def mmu_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        max_text_len = self.max_text_len - 1
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.pad_id] * (max_text_len - len(temp_ids))
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3) + [0] * (
                    max_text_len - len(temp_ids)
                )
            else:
                # should add the eos token
                temp_ids = temp_ids[: max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
                temp_masks = [1] * (
                    len(temp_ids) + image_ids.shape[-1] + 3
                )  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat(
                [
                    torch.tensor([self.ignore_id]).to(device),
                    torch.tensor([self.ignore_id]).to(device),
                    torch.ones_like(image_ids[i]) * self.ignore_id,
                    torch.tensor([self.ignore_id]).to(device),
                    torch.tensor(temp_ids).to(device),
                ],
                dim=0,
            )

            temp_label_ids = torch.where(
                temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids
            )

            temp_ids = torch.cat(
                [
                    self.sptids_dict["<|mmu|>"].to(device),  # task token
                    self.sptids_dict["<|soi|>"].to(device),
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"].to(device),
                    torch.tensor(temp_ids).to(device),
                ],
                dim=0,
            )

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(attention_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def __call__(self, input, task, padding=True, config=None):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "pre":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.pre_prompt(text_ids, image_ids, input[2])

        elif task == "pre_gen":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.pre_gen_prompt(text_ids, image_ids)

        elif task == "lm":
            text_ids = self.text_tokenizer(input[0], truncation=True)[
                "input_ids"
            ]  # (B, max_len)
            sequence_ids_with_masks = self.lm_prompt(text_ids, input[1])

        elif task == "mmu":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])["input_ids"]
            sequence_ids_with_masks = self.mmu_prompt(image_ids, text_ids)

            raise NotImplementedError

        return sequence_ids_with_masks


def create_attention_mask_for_mmu_vit(
    sequence,
    return_inverse_mask=True,
    system_prompt_len=0,
):
    N, L, H = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(
        sequence.device
    )
    index = 1 + system_prompt_len + 1 + 576
    # PART OF SYSTEM PROMPT SHOULD BE CAUSAL ALSO
    # causal_mask[:, :, :, :index] = 1
    causal_mask[:, :, :, 1 + system_prompt_len + 1 : index] = 1
    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(torch.int64)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(torch.int64).min
        )
        return inverted_mask
    else:
        return causal_mask


def create_attention_mask_predict_next_for_future_prediction(
    sequence,
    pad_id=128256,
    soi_id=128257,
    eoi_id=128258,
    rm_pad_in_image=False,
    return_inverse_mask=True,
):
    """
    UP-VLA의 혼합 어텐션 마스크를 생성하는 함수.

    핵심 아이디어:
      - 텍스트/PAD/<lvg> 위치: 인과적(causal) 어텐션 — 과거 토큰만 참조
      - 이미지(<|soi|>~<|eoi|>) 위치: 양방향(bidirectional) 어텐션 — 모든 이미지 토큰 참조

    시퀀스 예시 (L = max_text_len + 1 + num_vq + 1 + future_steps):
      [PAD...][<|t2i|>][SOT][텍스트][EOT] [<|soi|>][이미지 VQ 256개][<|eoi|>] [<|lvg|>×act_step]
      ←────────────── 텍스트 영역 ──────────→ ←── 이미지 영역(양방향) ──→ ←── 액션 영역 ───→

    반환: inverted_mask (return_inverse_mask=True) — -inf로 채워진 마스크 (어텐션 소프트맥스에 직접 더함)
          또는 bool 마스크 (return_inverse_mask=False)
    """
    # no change from original
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # ── 토큰 종류 식별 ──────────────────────────────────────────────
    is_padding = sequence == pad_id  # PAD 위치
    is_start_image = sequence == soi_id  # <|soi|> 위치
    is_end_image = sequence == eoi_id  # <|eoi|> 위치

    # <|soi|>와 <|eoi|>의 누적합으로 이미지 영역 탐지
    # cumsum(soi) > cumsum(eoi) 이면 soi 이후 eoi 이전 → 이미지 토큰
    # is_start_image, is_end_image 자체도 이미지 영역에 포함
    cumulative_start = torch.cumsum(is_start_image, dim=1)
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (
        (cumulative_start > cumulative_end) | is_start_image | is_end_image
    )  # 1+1024+1 (soi + 이미지 토큰 + eoi)

    is_text = ~(in_image_segment)  # 텍스트(PAD 포함) + <lvg> 위치 = 이미지 영역 반전

    # ── 기본 마스크 구성 ────────────────────────────────────────────
    # causal_mask: 하삼각 행렬 — 위치 j에서 위치 k(k≤j)만 참조 가능
    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    # mask_text: 텍스트 행(row)은 causal, 이미지 행(row)은 일단 causal로 초기화
    mask_text = (
        is_text[:, :, None] * causal_mask[None, :, :]
    )  # [B, L, L] — is_text인 행만 causal 마스크 적용

    # mask_text_image_bi: 이미지 영역의 모든 토큰이 서로 참조 가능 (양방향)
    is_text_image = is_text | in_image_segment  # 텍스트+이미지 (PAD 제외하지 않음)
    mask_text_image_bi = (
        is_text_image[:, :, None] * is_text_image[:, None, :]
    )  # [B, L, L] — 텍스트+이미지 영역의 outer product

    if rm_pad_in_image:
        # PAD가 이미지 어텐션에서 Key로 사용되지 않도록 제거 (선택적)
        sid_img = torch.where(sequence == soi_id)[1]
        for i in range(mask_text_image_bi.shape[0]):
            pad_end_idx = torch.where(sequence[i] == pad_id)
            if len(pad_end_idx[0]) != 0:
                pad_end_idx = pad_end_idx[0][-1]
                mask_text[i][pad_end_idx + 1 :, : pad_end_idx + 1] = 0
            id_padding = torch.where(is_padding[i] == True)
            mask_text_image_bi[i][sid_img[i] :, id_padding[0]] = 0

    # ── 혼합 마스크 최종 조립 ────────────────────────────────────────
    # 이미지 영역 행(in_image_segment)에만 양방향 마스크로 덮어씀
    # 텍스트/<lvg> 행은 mask_text(causal) 유지
    mask_text[in_image_segment] = mask_text_image_bi[in_image_segment]

    if return_inverse_mask:
        # 어텐션 소프트맥스에 더할 inverted mask: 참조 불가 위치 = -inf, 참조 가능 = 0
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1)  # [B, 1, L, L] — head 차원 추가
    else:
        return mask_text.unsqueeze(1)  # [B, 1, L, L]


if __name__ == "__main__":
    pass
