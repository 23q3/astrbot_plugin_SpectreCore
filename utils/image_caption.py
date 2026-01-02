from astrbot.api.all import *
from typing import Optional
import asyncio

class ImageCaptionUtils:
    """
    图片转述工具类

    用于调用大语言模型将图片转述为文本描述
    """

    # 保存context和config对象的静态变量
    context: Optional[Context] = None
    config: Optional[AstrBotConfig] = None
    # 图片描述缓存
    caption_cache: dict[str, str] = {}
    
    @staticmethod
    def init(context: Context, config: AstrBotConfig):
        """初始化图片转述工具类，保存context和config引用"""
        ImageCaptionUtils.context = context
        ImageCaptionUtils.config = config
    
    @staticmethod
    async def generate_image_caption(
            image: str, # 图片的base64编码或URL
            timeout: int = 30
        ) -> Optional[str]:
        """
        为单张图片生成文字描述
        
        Args:
            image: 图片的base64编码或URL
            timeout: 超时时间（秒）
            
        Returns:
            生成的图片描述文本，如果失败则返回None
        """
        # 检查缓存
        if image in ImageCaptionUtils.caption_cache:
            logger.debug(f"命中图片描述缓存: {image[:50]}...")
            return ImageCaptionUtils.caption_cache[image]
            
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

        provider_id = image_processing_config.get("image_caption_provider_id", "")
        # 获取提供商
        if provider_id == "":
            provider = context.get_using_provider()
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
            
            # 缓存结果
            if caption:
                 ImageCaptionUtils.caption_cache[image] = caption
                 logger.debug(f"缓存图片描述: {image[:50]}... -> {caption}")
                 
            return caption
        except asyncio.TimeoutError:
            logger.warning(f"图片转述超时，超过了{timeout}秒")
            return None
        except Exception as e:
            logger.error(f"图片转述失败: {e}")
            return None
