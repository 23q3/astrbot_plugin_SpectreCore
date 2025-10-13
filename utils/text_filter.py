"""
文本过滤工具，用于清理大模型回复中的额外内容
"""
from astrbot.api.all import logger

class TextFilter:
    """
    文本过滤工具类
    用于清理大模型回复中的额外内容，如思考过程
    """
    
    @staticmethod
    def filter_thinking_process(text: str) -> str:
        """
        过滤掉文本中被 <think></think> 标签包裹的思考过程内容
        
        Args:
            text: 输入文本
            
        Returns:
            过滤后的文本
        """
        if not text:
            return text
        
        try:
            # 检查文本开头是否有思考标签
            if text.startswith('<think>'):
                # 查找结束标签位置
                end_tag_pos = text.find('</think>')
                if end_tag_pos != -1:
                    # 移除开头到</think>结束标签的内容
                    return text[end_tag_pos + 8:].lstrip()  # 8是</think>的长度
                    
            return text
        except Exception as e:
            logger.error(f"过滤思考过程时出错: {e}")
            # 出错时返回原始文本
            return text

    @staticmethod
    def process_model_text(reply_text: str, config: dict) -> str:
        """
        处理模型的回复内容，根据配置进行文本过滤

        Args:
            reply_text: 模型原始回复文本
            config: 插件配置字典，包含filter_thinking和read_air等配置项

        Returns:
            处理后的回复文本
        """
        
        # 如果回复为空，直接返回
        if not reply_text:
            logger.warning("收到空回复")
            return reply_text
        
        filter_thinking_text = TextFilter.filter_thinking_process(reply_text)
        # 读空气
        if config.get('read_air', False):
            # 检查是否包含<NO_RESPONSE>标记
            if "<NO_RESPONSE>" in filter_thinking_text:
                return "<NO_RESPONSE>"
            
        # 思考过程过滤
        if config.get('filter_thinking', True):
            if filter_thinking_text != reply_text:
                logger.debug("已过滤思考过程")
                return filter_thinking_text
        
        logger.debug("未进行任何过滤")
        return reply_text 