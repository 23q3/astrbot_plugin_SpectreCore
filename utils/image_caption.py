from astrbot.api.all import *
from typing import Optional
import asyncio
import base64
import binascii
import os
import re
import urllib.parse
import urllib.request
import urllib.error
from .image_cache import ImageCacheManager

class ImageCaptionUtils:
    """
    图片转述工具类

    用于调用大语言模型将图片转述为文本描述
    """

    # 保存context和config对象的静态变量
    context: Optional[Context] = None
    config: Optional[AstrBotConfig] = None
    DEFAULT_FAILED_IMAGE_SKIP_WINDOW_SECONDS = 300
    SAFE_NETLOC_LABEL_RE = re.compile(r"[A-Za-z0-9_-]+")
    
    @staticmethod
    def init(context: Context, config: AstrBotConfig):
        """初始化图片转述工具类，保存context和config引用"""
        ImageCaptionUtils.context = context
        ImageCaptionUtils.config = config
        # 初始化图片缓存管理器
        ImageCacheManager.init(config)
    
    @staticmethod
    def get_failed_image_skip_window_seconds() -> int:
        """
        获取失败图片跳过策略的时间窗口（秒）
        """
        config = ImageCaptionUtils.config
        if not config:
            return ImageCaptionUtils.DEFAULT_FAILED_IMAGE_SKIP_WINDOW_SECONDS

        image_processing_config = config.get("image_processing", {})
        skip_window_seconds = image_processing_config.get(
            "failed_image_skip_window_seconds",
            ImageCaptionUtils.DEFAULT_FAILED_IMAGE_SKIP_WINDOW_SECONDS
        )
        if not isinstance(skip_window_seconds, int) or skip_window_seconds < 0:
            return ImageCaptionUtils.DEFAULT_FAILED_IMAGE_SKIP_WINDOW_SECONDS
        return skip_window_seconds

    @staticmethod
    def _check_url_accessible(url: str, timeout: int) -> bool:
        """
        同步检查图片 URL 是否可访问（供异步线程调用）
        """
        head_fallback_statuses = {
            400,  # Bad Request（部分代理/服务不支持 HEAD）
            405,  # Method Not Allowed
            501,  # Not Implemented
        }
        range_fallback_statuses = {
            400,  # Bad Request（部分服务不支持 Range）
            416,  # Range Not Satisfiable
        }
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if not (200 <= status < 400):
                    return False
                content_length = resp.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        if int(content_length) <= 0:
                            return False
                    except (TypeError, ValueError):
                        pass
                return True
        except urllib.error.HTTPError as e:
            if e.code not in head_fallback_statuses:
                return False
        except Exception:
            return False

        try:
            req = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if not (200 <= status < 400):
                    return False
                return True
        except urllib.error.HTTPError as e:
            if e.code not in range_fallback_statuses:
                return False
        except Exception:
            return False

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if not (200 <= status < 400):
                    return False
                return True
        except Exception:
            return False

    @staticmethod
    def _check_local_image_accessible(image_path: str) -> bool:
        """
        同步检查本地图片是否存在且可读取（供异步线程调用）
        """
        try:
            if not image_path:
                return False
            if not os.path.exists(image_path) or not os.path.isfile(image_path):
                return False
            with open(image_path, "rb") as f:
                return bool(f.read(1))
        except Exception:
            return False

    @staticmethod
    def _is_safe_file_netloc(netloc: str) -> bool:
        """
        校验 file:// 的 netloc 是否安全（仅允许主机名格式）
        """
        if not netloc or len(netloc) > 253:
            return False
        labels = netloc.split(".")
        for label in labels:
            if not label or len(label) > 63:
                return False
            if label.startswith("-") or label.endswith("-"):
                return False
            if not ImageCaptionUtils.SAFE_NETLOC_LABEL_RE.fullmatch(label):
                return False
        return True

    @staticmethod
    def _is_safe_unc_path(path: str) -> bool:
        """
        校验 UNC 路径是否包含可疑的路径穿越片段
        """
        if not path:
            return False
        if ":" in path:
            return False
        normalized = path.replace("\\", "/")
        return ".." not in normalized.split("/")

    @staticmethod
    async def _ensure_image_accessible(image: str, timeout: int) -> bool:
        """
        确保图片存在且可获取

        注意：file:// 的网络路径仅在 Windows 下支持，其他平台会直接拒绝。
        """
        if not image:
            return False

        if image.startswith("http://") or image.startswith("https://"):
            return await asyncio.to_thread(ImageCaptionUtils._check_url_accessible, image, timeout)

        if image.startswith("file://"):
            try:
                parsed = urllib.parse.urlparse(image)
                if parsed.netloc and parsed.netloc not in ("", "localhost"):
                    if os.name == "nt":
                        if not ImageCaptionUtils._is_safe_file_netloc(parsed.netloc):
                            logger.warning(f"不安全的 file:// 网络地址: {image}")
                            return False
                        unc_path = urllib.request.url2pathname(parsed.path or "")
                        if not ImageCaptionUtils._is_safe_unc_path(unc_path):
                            logger.warning(f"不安全的 file:// UNC 路径: {image}")
                            return False
                        image_path = f"\\\\{parsed.netloc}{unc_path}"
                    else:
                        logger.warning(f"不支持的 file:// 网络路径: {image}")
                        return False
                else:
                    image_path = urllib.request.url2pathname(parsed.path or "")
                if not image_path:
                    return False
                return await asyncio.to_thread(ImageCaptionUtils._check_local_image_accessible, image_path)
            except Exception:
                return False

        expanded_path = os.path.expanduser(image)
        if os.path.exists(expanded_path):
            return await asyncio.to_thread(ImageCaptionUtils._check_local_image_accessible, expanded_path)
        if image.startswith("~"):
            # 展开后的路径不同，视为用户路径而非 base64
            return False

        if image.startswith("data:"):
            try:
                header, b64data = image.split(",", 1)
                if "base64" not in header:
                    return False
                base64.b64decode(b64data, validate=True)
                return True
            except (ValueError, binascii.Error):
                return False

        # 普通 base64 字符串
        try:
            base64.b64decode(image, validate=True)
            return True
        except (binascii.Error, ValueError):
            return False

    @staticmethod
    async def generate_image_caption(
            image: str, # 图片的base64编码或URL
            umo: Optional[str] = None, # unified_msg_origin，用于 UMO 路由
            timeout: int = 30,
            latest_success_timestamp: Optional[float] = None
        ) -> Optional[str]:
        """
        为单张图片生成文字描述

        Args:
            image: 图片的base64编码或URL
            umo: unified_msg_origin，用于获取对应 UMO 的 provider
            timeout: 超时时间（秒）
            latest_success_timestamp: 最近一次成功转述时间戳（用于失败图片跳过策略）

        Returns:
            生成的图片描述文本，如果失败则返回None
        """
        # 检查持久化缓存
        cached_caption = ImageCacheManager.get(image)
        if cached_caption is not None:
            ImageCacheManager.clear_failed(image)
            logger.debug(f"命中图片描述缓存: {image[:50]}...")
            return cached_caption
            
        # 获取配置
        config = ImageCaptionUtils.config
        context = ImageCaptionUtils.context

        if not config or not context:
            logger.warning("ImageCaptionUtils 未初始化")
            return None

        # 检查是否已启用图片转述
        image_processing_config = config.get("image_processing", {})
        if not image_processing_config.get("use_image_caption", False):
            return None

        skip_window_seconds = ImageCaptionUtils.get_failed_image_skip_window_seconds()

        if ImageCacheManager.should_skip_failed_image(image, latest_success_timestamp, skip_window_seconds):
            logger.debug(f"跳过失败图片转述（该图片失败记录早于本轮最近一次成功，且时间间隔在窗口内）: {image[:50]}...")
            return None

        # 在调用大模型前确认图片可获取
        image_accessible = await ImageCaptionUtils._ensure_image_accessible(image, timeout=min(timeout, 10))
        if not image_accessible:
            logger.warning(f"图片无法获取或不存在，已跳过转述: {image[:50]}...")
            ImageCacheManager.set_failed(image)
            return None

        provider_id = image_processing_config.get("image_caption_provider_id", "")
        # 获取提供商，支持 UMO 路由
        if provider_id == "":
            provider = context.get_using_provider(umo=umo)
        else:
            provider = context.get_provider_by_id(provider_id)

        if not provider or not hasattr(provider, "text_chat"):
             logger.warning(f"无法找到提供商: {provider_id if provider_id else '默认'}")
             return None

        text_chat = getattr(provider, "text_chat")
        try:
            # 带超时控制的调用大模型进行图片转述
            async def call_llm():
                return await text_chat(
                    prompt=image_processing_config.get("image_caption_prompt", "请直接简短描述这张图片"),
                    contexts=[],
                    image_urls=[image],
                    func_tool=None,
                    system_prompt=""
                )
            
            # 使用asyncio.wait_for添加超时控制
            llm_response = await asyncio.wait_for(call_llm(), timeout=timeout)
            caption = llm_response.completion_text
            
            # 缓存结果到持久化缓存
            if caption:
                 ImageCacheManager.set(image, caption)
                 ImageCacheManager.clear_failed(image)
                 logger.debug(f"缓存到持久化存储: {image[:50]}...")
            else:
                 ImageCacheManager.set_failed(image)
                 
            return caption
        except asyncio.TimeoutError:
            logger.warning(f"图片转述超时，超过了{timeout}秒")
            ImageCacheManager.set_failed(image)
            return None
        except Exception as e:
            logger.error(f"图片转述失败: {e}")
            ImageCacheManager.set_failed(image)
            return None
