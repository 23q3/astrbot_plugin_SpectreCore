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
    
    # 常量定义
    MAX_RETENTION_DAYS = 365
    DEFAULT_RETENTION_DAYS = 7
    HOURS_PER_DAY = 24
    SECONDS_PER_HOUR = 3600
    WRITE_THRESHOLD = 10  # 每10次写入保存一次
    
    # 保存配置对象的静态变量
    config: Optional[AstrBotConfig] = None
    # 基础存储路径
    base_storage_path: Optional[str] = None
    # 内存缓存（用于快速查询）
    memory_cache: Dict[str, tuple[str, float]] = {}
    # 失败记录缓存（hash -> failure_timestamp）
    failure_cache: Dict[str, float] = {}
    # 记录写入次数，用于周期性保存
    write_count: int = 0
    
    @staticmethod
    def init(config: AstrBotConfig):
        """
        初始化图片缓存管理器，保存config引用
        
        Args:
            config: AstrBotConfig 对象
        """
        ImageCacheManager.config = config
        ImageCacheManager.write_count = 0  # 重置写入计数
        ImageCacheManager.memory_cache.clear()  # 清空内存缓存，确保从磁盘重新加载
        ImageCacheManager.failure_cache.clear()  # 清空失败缓存，确保从磁盘重新加载
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
            
            # 兼容两种格式：
            # 1) 旧格式: {hash: [caption, timestamp]}
            # 2) 新格式: {"captions": {...}, "failures": {...}}
            caption_data = cache_data
            failure_data = {}
            if isinstance(cache_data, dict) and ("captions" in cache_data or "failures" in cache_data):
                caption_data = cache_data.get("captions", {})
                failure_data = cache_data.get("failures", {})

            # 加载成功缓存到内存
            if isinstance(caption_data, dict):
                for key, value in caption_data.items():
                    try:
                        # 要求恰好2个元素
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            caption, timestamp = value[0], value[1]
                            # 验证类型
                            if isinstance(caption, str) and isinstance(timestamp, (int, float)):
                                ImageCacheManager.memory_cache[key] = (caption, timestamp)
                            else:
                                logger.warning(f"缓存条目类型不正确，跳过: {key}")
                        else:
                            logger.warning(f"缓存条目格式不正确，跳过: {key}")
                    except Exception as e:
                        logger.warning(f"加载缓存条目失败 {key}: {e}")
                        
                logger.info(f"成功从磁盘加载 {len(ImageCacheManager.memory_cache)} 条图片缓存")
            else:
                logger.warning(f"图片缓存数据格式不正确，跳过加载")

            # 加载失败缓存到内存
            if isinstance(failure_data, dict):
                for key, value in failure_data.items():
                    if isinstance(value, (int, float)):
                        ImageCacheManager.failure_cache[key] = float(value)
                    else:
                        logger.warning(f"失败缓存条目格式不正确，跳过: {key}")
            elif failure_data:
                logger.warning("失败缓存数据格式不正确，跳过加载")
                 
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
            
            # 转换内存缓存格式为可序列化的格式，并验证条目
            serializable_cache = {}
            skipped_count = 0
            for key, value in ImageCacheManager.memory_cache.items():
                if isinstance(value, tuple) and len(value) == 2:
                    caption, timestamp = value
                    # 验证条目有效性
                    if isinstance(caption, str) and isinstance(timestamp, (int, float)):
                        serializable_cache[key] = [caption, timestamp]
                    else:
                        skipped_count += 1
                        logger.debug(f"跳过格式不正确的缓存条目: {key}")
                else:
                    skipped_count += 1
                    logger.debug(f"跳过格式不正确的缓存条目: {key}")
            
            serializable_failures = {}
            for key, value in ImageCacheManager.failure_cache.items():
                if isinstance(value, (int, float)):
                    serializable_failures[key] = float(value)
                else:
                    logger.debug(f"跳过格式不正确的失败缓存条目: {key}")

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "captions": serializable_cache,
                        "failures": serializable_failures
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            
            if skipped_count > 0:
                logger.debug(f"成功保存 {len(serializable_cache)} 条有效缓存到磁盘，跳过 {skipped_count} 条格式不正确的条目")
            else:
                logger.debug(f"成功保存 {len(serializable_cache)} 条图片缓存到磁盘")
            
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
                
                # 统一处理缓存数据格式（要求严格的tuple/list格式，恰好包含2个元素）
                if isinstance(cached_data, (tuple, list)) and len(cached_data) == 2:
                    caption, timestamp = cached_data[0], cached_data[1]
                    # 验证提取的值类型
                    if not isinstance(caption, str):
                        logger.warning(f"缓存条目格式不正确，期望字符串但获得 {type(caption).__name__}")
                        return None
                    if not isinstance(timestamp, (int, float)):
                        logger.warning(f"缓存条目时间戳格式不正确，期望数字但获得 {type(timestamp).__name__}")
                        return None
                else:
                    logger.warning(f"缓存条目格式不正确，期望恰好2个元素的tuple/list但获得 {type(cached_data).__name__}")
                    return None
                
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
            
            # 基于阈值的周期性保存（更稳定，避免过度I/O）
            ImageCacheManager.write_count += 1
            if ImageCacheManager.write_count >= ImageCacheManager.WRITE_THRESHOLD:
                ImageCacheManager._save_cache_to_disk()
                ImageCacheManager.write_count = 0
            
            logger.debug(f"缓存图片描述: {image[:50]}...")
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
            ImageCacheManager.failure_cache.clear()
            ImageCacheManager.write_count = 0
            
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
            retention_days = image_processing_config.get("image_retention_days", ImageCacheManager.DEFAULT_RETENTION_DAYS)
            
            # 验证配置值有效性
            if retention_days < 1 or retention_days > ImageCacheManager.MAX_RETENTION_DAYS:
                logger.warning(f"图片保留天数配置无效: {retention_days}，使用默认值{ImageCacheManager.DEFAULT_RETENTION_DAYS}天")
                retention_days = ImageCacheManager.DEFAULT_RETENTION_DAYS
            
            current_time = time.time()
            cleanup_threshold = retention_days * ImageCacheManager.HOURS_PER_DAY * ImageCacheManager.SECONDS_PER_HOUR
            removed_count = 0
            
            keys_to_remove = []
            for key, value in ImageCacheManager.memory_cache.items():
                # 统一处理所有格式的缓存条目，要求恰好2个元素
                timestamp = None
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    timestamp = value[1]
                
                # 如果没有有效的时间戳，视为损坏的条目，标记删除
                if timestamp is None:
                    keys_to_remove.append(key)
                    removed_count += 1
                    logger.debug(f"删除时间戳无效的缓存条目: {key}")
                # 检查是否超过保留期限
                elif current_time - timestamp > cleanup_threshold:
                    keys_to_remove.append(key)
                    removed_count += 1
            
            for key in keys_to_remove:
                del ImageCacheManager.memory_cache[key]

            failure_keys_to_remove = []
            for key, timestamp in ImageCacheManager.failure_cache.items():
                if not isinstance(timestamp, (int, float)):
                    failure_keys_to_remove.append(key)
                    removed_count += 1
                elif current_time - timestamp > cleanup_threshold:
                    failure_keys_to_remove.append(key)
                    removed_count += 1

            for key in failure_keys_to_remove:
                del ImageCacheManager.failure_cache[key]
            
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

    @staticmethod
    def get_failed_timestamp(image: str) -> Optional[float]:
        """
        获取图片最近一次转述失败时间戳
        """
        try:
            image_hash = ImageCacheManager._generate_image_hash(image)
            timestamp = ImageCacheManager.failure_cache.get(image_hash)
            if isinstance(timestamp, (int, float)):
                return float(timestamp)
            return None
        except Exception as e:
            logger.error(f"获取失败记录失败: {e}")
            return None

    @staticmethod
    def is_failed(image: str) -> bool:
        """
        判断图片是否有失败记录
        """
        return ImageCacheManager.get_failed_timestamp(image) is not None

    @staticmethod
    def set_failed(image: str) -> bool:
        """
        记录图片转述失败
        """
        try:
            image_hash = ImageCacheManager._generate_image_hash(image)
            ImageCacheManager.failure_cache[image_hash] = time.time()

            ImageCacheManager.write_count += 1
            if ImageCacheManager.write_count >= ImageCacheManager.WRITE_THRESHOLD:
                ImageCacheManager._save_cache_to_disk()
                ImageCacheManager.write_count = 0

            return True
        except Exception as e:
            logger.error(f"记录失败缓存失败: {e}")
            return False

    @staticmethod
    def clear_failed(image: str) -> bool:
        """
        清理图片失败记录
        """
        try:
            image_hash = ImageCacheManager._generate_image_hash(image)
            if image_hash in ImageCacheManager.failure_cache:
                del ImageCacheManager.failure_cache[image_hash]

                ImageCacheManager.write_count += 1
                if ImageCacheManager.write_count >= ImageCacheManager.WRITE_THRESHOLD:
                    ImageCacheManager._save_cache_to_disk()
                    ImageCacheManager.write_count = 0

            return True
        except Exception as e:
            logger.error(f"清理失败缓存失败: {e}")
            return False

    @staticmethod
    def should_skip_failed_image(image: str, latest_success_timestamp: Optional[float], window_seconds: int) -> bool:
        """
        判断失败图片是否应跳过转述：
        - 存在失败记录
        - 失败时间早于最近成功时间
        - 且二者间隔在窗口时间内
        """
        if latest_success_timestamp is None or window_seconds <= 0:
            return False

        failed_timestamp = ImageCacheManager.get_failed_timestamp(image)
        if failed_timestamp is None:
            return False

        return failed_timestamp <= latest_success_timestamp and (latest_success_timestamp - failed_timestamp) <= window_seconds
