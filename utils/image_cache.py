import os
import json
import time
import hashlib
import traceback
from typing import Optional, Dict
from astrbot.api.all import *


class ImageCacheManager:
    """
    图片转述缓存管理器
    
    用于持久化存储图片转述缓存，避免重复的图片转述请求
    """
    
    # 保存配置对象的静态变量
    config: Optional[AstrBotConfig] = None
    # 基础存储路径
    base_storage_path: Optional[str] = None
    # 内存缓存（用于快速查询）
    memory_cache: Dict[str, tuple[str, float]] = {}
    
    @staticmethod
    def init(config: AstrBotConfig):
        """
        初始化图片缓存管理器，保存config引用
        
        Args:
            config: AstrBotConfig 对象
        """
        ImageCacheManager.config = config
        # 初始化基础存储路径
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path
        astrbot_data_path = get_astrbot_data_path()
        ImageCacheManager.base_storage_path = os.path.join(astrbot_data_path, "data", "image_caption_cache")
        ImageCacheManager._ensure_dir(ImageCacheManager.base_storage_path)
        logger.info(f"图片缓存存储路径初始化: {ImageCacheManager.base_storage_path}")
        
        # 加载现有的缓存到内存
        ImageCacheManager._load_cache_from_disk()
    
    @staticmethod
    def _ensure_dir(directory: str) -> None:
        """确保目录存在，不存在则创建"""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def _get_cache_file_path() -> str:
        """获取缓存文件路径"""
        if not ImageCacheManager.base_storage_path:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path
            astrbot_data_path = get_astrbot_data_path()
            ImageCacheManager.base_storage_path = os.path.join(astrbot_data_path, "data", "image_caption_cache")
            ImageCacheManager._ensure_dir(ImageCacheManager.base_storage_path)
        
        return os.path.join(ImageCacheManager.base_storage_path, "caption_cache.json")
    
    @staticmethod
    def _generate_image_hash(image: str) -> str:
        """
        为图片生成哈希值（用于作为缓存键）
        
        使用 SHA256 生成固定长度的哈希，避免过长的键名
        
        Args:
            image: 图片的base64编码或URL
            
        Returns:
            图片的哈希值
        """
        return hashlib.sha256(image.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _load_cache_from_disk() -> None:
        """从磁盘加载缓存到内存"""
        try:
            cache_file = ImageCacheManager._get_cache_file_path()
            
            if not os.path.exists(cache_file):
                logger.debug("缓存文件不存在，跳过加载")
                return
            
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # 加载缓存到内存
            if isinstance(cache_data, dict):
                ImageCacheManager.memory_cache = cache_data
                logger.info(f"成功从磁盘加载 {len(cache_data)} 条图片缓存")
            else:
                logger.warning(f"缓存文件格式不正确，跳过加载")
                
        except Exception as e:
            logger.error(f"从磁盘加载缓存失败: {e}")
            logger.debug(traceback.format_exc())
    
    @staticmethod
    def _save_cache_to_disk() -> None:
        """将内存缓存保存到磁盘"""
        try:
            cache_file = ImageCacheManager._get_cache_file_path()
            
            # 确保父目录存在
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # 转换内存缓存格式为可序列化的格式
            serializable_cache = {}
            for key, value in ImageCacheManager.memory_cache.items():
                if isinstance(value, tuple) and len(value) == 2:
                    caption, timestamp = value
                    serializable_cache[key] = [caption, timestamp]
                else:
                    serializable_cache[key] = value
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable_cache, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"成功保存 {len(ImageCacheManager.memory_cache)} 条图片缓存到磁盘")
            
        except Exception as e:
            logger.error(f"保存缓存到磁盘失败: {e}")
            logger.debug(traceback.format_exc())
    
    @staticmethod
    def get(image: str) -> Optional[str]:
        """
        获取缓存的图片转述
        
        Args:
            image: 图片的base64编码或URL
            
        Returns:
            缓存的转述文本，如果不存在则返回None
        """
        try:
            image_hash = ImageCacheManager._generate_image_hash(image)
            
            if image_hash in ImageCacheManager.memory_cache:
                cached_data = ImageCacheManager.memory_cache[image_hash]
                
                # 处理缓存数据格式
                if isinstance(cached_data, tuple) and len(cached_data) >= 1:
                    caption = cached_data[0]
                elif isinstance(cached_data, list) and len(cached_data) >= 1:
                    caption = cached_data[0]
                else:
                    caption = cached_data
                
                logger.debug(f"命中图片描述缓存: {image[:50]}...")
                return caption
            
            return None
            
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None
    
    @staticmethod
    def set(image: str, caption: str) -> bool:
        """
        存储图片转述到缓存
        
        Args:
            image: 图片的base64编码或URL
            caption: 图片的转述文本
            
        Returns:
            是否存储成功
        """
        try:
            image_hash = ImageCacheManager._generate_image_hash(image)
            
            # 存储为元组 (caption, timestamp) 用于后续清理
            ImageCacheManager.memory_cache[image_hash] = (caption, time.time())
            
            # 异步保存到磁盘（通过随机概率减少I/O）
            import random
            if random.random() < 0.2:  # 20% 的概率保存一次
                ImageCacheManager._save_cache_to_disk()
            
            logger.debug(f"缓存图片描述: {image[:50]}... -> {caption}")
            return True
            
        except Exception as e:
            logger.error(f"存储缓存失败: {e}")
            return False
    
    @staticmethod
    def clear() -> bool:
        """
        清空所有缓存
        
        Returns:
            是否清空成功
        """
        try:
            ImageCacheManager.memory_cache.clear()
            
            cache_file = ImageCacheManager._get_cache_file_path()
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            logger.info("已清空所有图片缓存")
            return True
            
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False
    
    @staticmethod
    def cleanup_old_entries() -> None:
        """
        清理超过配置天数的缓存条目
        
        防止缓存无限增长
        """
        try:
            if not ImageCacheManager.config:
                logger.debug("配置未初始化，跳过缓存清理")
                return
            
            image_processing_config = ImageCacheManager.config.get("image_processing", {})
            retention_days = image_processing_config.get("image_retention_days", 7)
            
            if retention_days < 1 or retention_days > 365:
                logger.warning(f"图片保留天数配置无效: {retention_days}，使用默认值7天")
                retention_days = 7
            
            current_time = time.time()
            cleanup_threshold = retention_days * 24 * 3600  # 配置的天数转换为秒
            removed_count = 0
            
            keys_to_remove = []
            for key, value in ImageCacheManager.memory_cache.items():
                if isinstance(value, tuple) and len(value) == 2:
                    caption, timestamp = value
                    if current_time - timestamp > cleanup_threshold:
                        keys_to_remove.append(key)
                        removed_count += 1
            
            for key in keys_to_remove:
                del ImageCacheManager.memory_cache[key]
            
            if removed_count > 0:
                logger.info(f"清理过期缓存完成，清理了 {removed_count} 条超过 {retention_days} 天的缓存条目")
                # 清理后保存一次
                ImageCacheManager._save_cache_to_disk()
            
        except Exception as e:
            logger.error(f"清理缓存时发生错误: {e}")
    
    @staticmethod
    def force_save() -> None:
        """强制将缓存保存到磁盘（用于关闭前调用）"""
        try:
            ImageCacheManager._save_cache_to_disk()
        except Exception as e:
            logger.error(f"强制保存缓存失败: {e}")
