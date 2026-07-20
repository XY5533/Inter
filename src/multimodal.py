"""
多模态处理器 — 语音识别、图片理解、视频抽帧

所有模态统一转为文本后，再进入 ReAct + RAG 管线。
使用 DashScope 原生 API，与现有技术栈一致。
"""

import os
import io
import base64
import tempfile
import re
from typing import List, Optional
from pathlib import Path

import dashscope
from dashscope import MultiModalConversation

import config as global_config
from prompt_templates import IMAGE_DESCRIPTION_PROMPT, VIDEO_SUMMARY_PROMPT


class MultimodalProcessor:
    """多模态输入处理器

    使用方式：
        mp = MultimodalProcessor()
        text = mp.transcribe_audio(audio_bytes)     # 语音 → 文本
        desc = mp.describe_image(image_bytes)        # 图片 → 描述
        frames = mp.extract_video_frames(video_bytes)  # 视频 → 帧列表
        summary = mp.summarize_video(video_bytes)    # 视频 → 文本摘要
    """

    def __init__(self, api_key: str = None, vision_model: str = None):
        self.api_key = api_key or global_config.DASHSCOPE_API_KEY
        if self.api_key:
            dashscope.api_key = self.api_key
        self.vision_model = vision_model or global_config.VISION_MODEL

    # ================================================================
    # 语音识别（转文本）
    # ================================================================

    def transcribe_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """将音频转为文本

        Args:
            audio_bytes: 音频文件二进制数据 (wav/mp3)
            sample_rate: 采样率（默认 16000）

        Returns:
            识别的文本，失败返回空字符串

        使用 DashScope 的 paraformer-v2 语音识别模型。
        如果 paraformer 不可用，降级到 qwen-audio 模型。
        """
        if not self.api_key:
            return "[错误] 请先配置 DASHSCOPE_API_KEY"

        # 将音频字节写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # 方案1: 使用 paraformer-v2 语音识别 API
            task_response = dashscope.audio.asr.Transcription.async_call(
                model="paraformer-v2",
                file_urls=[f"file://{tmp_path}"],
                sample_rate=sample_rate,
            )
            # 等待并获取结果
            import time as _time
            max_wait = 60  # 最长等待 60 秒
            for _ in range(max_wait * 2):
                result = dashscope.audio.asr.Transcription.wait(task_response)
                if result.status_code == 200:
                    # 提取文本
                    if (result.output and result.output.get("results") and
                            result.output["results"]):
                        transcript = ""
                        for item in result.output["results"]:
                            if item.get("sentences"):
                                for sent in item["sentences"]:
                                    transcript += sent.get("text", "")
                        if transcript:
                            return transcript
                    break
                elif result.status_code != 200:
                    _time.sleep(0.5)
                    continue

            # 如果 paraformer 没结果，用占位返回
            return "[语音识别] 未能识别出文字内容，请重试或使用文本输入"

        except Exception as e:
            # 降级方案: 尝试用 qwen-audio 做语音理解
            try:
                return self._transcribe_with_qwen_audio(tmp_path)
            except Exception:
                return f"[语音识别失败] {e}"
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _transcribe_with_qwen_audio(self, audio_path: str) -> str:
        """降级方案：使用 qwen-audio 模型做语音转文本"""
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"audio": f"data:audio/wav;base64,{audio_base64}"},
                {"text": "请将这段音频转写为文字，只输出文字内容。"},
            ],
        }]

        response = MultiModalConversation.call(
            model="qwen-audio-turbo-latest",
            messages=messages,
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content[0].get("text", "")
        return ""

    # ================================================================
    # 图片理解
    # ================================================================

    def describe_image(self, image_bytes: bytes, context: str = "",
                       detail_prompt: str = None) -> str:
        """理解图片内容，返回文字描述

        Args:
            image_bytes: 图片二进制数据 (jpg/png)
            context: 可选的上下文引导（如"用户上传了一张产品照片"）
            detail_prompt: 自定义描述提示词

        Returns:
            图片的文字描述
        """
        if not self.api_key:
            return "[错误] 请先配置 DASHSCOPE_API_KEY"

        # Base64 编码图片
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 检测图片格式
        magic = image_bytes[:4]
        if magic[:2] == b'\xff\xd8':
            mime = "image/jpeg"
        elif magic[:4] == b'\x89PNG':
            mime = "image/png"
        else:
            mime = "image/jpeg"  # 默认

        prompt_text = detail_prompt or IMAGE_DESCRIPTION_PROMPT
        if context:
            prompt_text = f"[上下文: {context}]\n\n{prompt_text}"

        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:{mime};base64,{image_base64}"},
                {"text": prompt_text},
            ],
        }]

        try:
            response = MultiModalConversation.call(
                model=self.vision_model,
                messages=messages,
            )
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                # 可能是列表或字符串
                if isinstance(content, list):
                    return content[0].get("text", str(content))
                return str(content)
            else:
                return f"[图片描述失败] HTTP {response.status_code}: {response.message}"
        except Exception as e:
            return f"[图片描述失败] {e}"

    def answer_with_image(self, image_bytes: bytes, question: str,
                          knowledge_context: str = "") -> str:
        """结合图片内容回答问题（直接调用 qwen-vl，不经过 RAG）

        Args:
            image_bytes: 图片二进制数据
            question: 用户问题
            knowledge_context: 可选的知识库检索结果（从 RAG 管线传入）

        Returns:
            综合回答文本
        """
        if not self.api_key:
            return "[错误] 请先配置 DASHSCOPE_API_KEY"

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        magic = image_bytes[:4]
        if magic[:2] == b'\xff\xd8':
            mime = "image/jpeg"
        elif magic[:4] == b'\x89PNG':
            mime = "image/png"
        else:
            mime = "image/jpeg"

        prompt_text = "请根据这张图片回答用户问题。"
        if knowledge_context:
            prompt_text += f"\n\n参考资料：\n{knowledge_context}"
        prompt_text += f"\n\n用户问题：{question}"

        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:{mime};base64,{image_base64}"},
                {"text": prompt_text},
            ],
        }]

        try:
            response = MultiModalConversation.call(
                model=self.vision_model,
                messages=messages,
            )
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                if isinstance(content, list):
                    return content[0].get("text", str(content))
                return str(content)
            else:
                return f"[图片问答失败] HTTP {response.status_code}"
        except Exception as e:
            return f"[图片问答失败] {e}"

    # ================================================================
    # 视频处理
    # ================================================================

    def extract_video_frames(self, video_bytes: bytes, fps: int = None) -> List[bytes]:
        """从视频中提取关键帧

        Args:
            video_bytes: 视频文件二进制数据
            fps: 每秒抽取帧数（默认 config.VIDEO_FPS）

        Returns:
            帧图片列表（每帧为 bytes）
        """
        if fps is None:
            fps = global_config.VIDEO_FPS

        frames = []

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            # 动态导入 opencv（可能未安装）
            import cv2

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return frames

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30  # 默认 30fps

            frame_interval = max(1, int(video_fps / fps))
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # 编码为 JPEG
                    success, encoded = cv2.imencode(".jpg", frame)
                    if success:
                        frames.append(encoded.tobytes())

                frame_count += 1
                # 限制最多 20 帧
                if len(frames) >= 20:
                    break

            cap.release()
        except ImportError:
            pass  # opencv 未安装时返回空列表
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return frames

    def summarize_video(self, video_bytes: bytes, fps: int = None) -> str:
        """将视频内容总结为文字

        Args:
            video_bytes: 视频文件二进制数据
            fps: 每秒抽取帧数

        Returns:
            视频内容文字摘要
        """
        frames = self.extract_video_frames(video_bytes, fps=fps)
        if not frames:
            return "[视频处理] 未能提取视频帧，可能需要安装 opencv-python"

        # 逐帧描述
        descriptions = []
        for i, frame_bytes in enumerate(frames):
            desc = self.describe_image(
                frame_bytes,
                context=f"视频第 {i+1}/{len(frames)} 帧",
            )
            if not desc.startswith("["):
                descriptions.append(f"帧{i+1}: {desc[:200]}")

        if not descriptions:
            return "[视频处理] 未能生成帧描述"

        # 用 LLM 汇总（如果 llm 已注入）
        if self._llm:
            try:
                from langchain_core.messages import HumanMessage
                prompt = VIDEO_SUMMARY_PROMPT.format(
                    frame_descriptions="\n".join(descriptions)
                )
                response = self._llm.invoke([HumanMessage(content=prompt)])
                return response.content if hasattr(response, "content") else str(response)
            except Exception:
                pass

        return "视频摘要:\n" + "\n".join(descriptions)

    # ================================================================
    # 集成方法
    # ================================================================

    _llm = None

    def set_llm(self, llm):
        """注入 LLM 实例（用于视频汇总）"""
        self._llm = llm

    def process_input(self, text: str = "", image_bytes: bytes = None,
                      audio_bytes: bytes = None, video_bytes: bytes = None) -> dict:
        """统一输入处理入口

        Returns:
            {
                "text": "合并后的文本输入",
                "image_description": "图片描述（可选）",
                "audio_transcript": "语音转写（可选）",
                "video_summary": "视频摘要（可选）",
            }
        """
        result = {"text": text or ""}

        if audio_bytes:
            transcript = self.transcribe_audio(audio_bytes)
            result["audio_transcript"] = transcript
            if transcript and not transcript.startswith("["):
                result["text"] = (result["text"] + " " + transcript).strip()

        if image_bytes:
            desc = self.describe_image(image_bytes)
            result["image_description"] = desc
            if desc and not desc.startswith("["):
                result["text"] += f"\n\n[图片描述] {desc}"

        if video_bytes:
            summary = self.summarize_video(video_bytes)
            result["video_summary"] = summary
            if summary and not summary.startswith("["):
                result["text"] += f"\n\n[视频摘要] {summary}"

        return result
