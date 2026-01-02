from astrbot.api.all import *
from astrbot.api.provider import Personality
from typing import Dict, List, Optional, Any

class PersonaUtils:
    """
    人格信息工具类
    用于获取和管理AstrBot中的人格信息
    """
    
    @staticmethod
    def get_all_personas(context: Context) -> List[Personality]:
        """
        获取所有已加载的人格
        
        Args:
            context: Context对象
            
        Returns:
            所有已加载的人格列表
        """
        try:
            return context.provider_manager.personas
        except Exception as e:
            logger.error(f"获取所有人格失败: {e}")
            return []
    
    @staticmethod
    def get_default_persona(context: Context) -> Optional[str]:
        """
        获取默认人格的ID
        
        Args:
            context: Context对象
            
        Returns:
            默认人格的ID，如果获取失败则返回None
        """
        try:
            persona = context.persona_manager.selected_default_persona_v3
            return persona["name"] if persona else None
        except Exception as e:
            logger.error(f"获取默认人格失败: {e}")
            return None
    
    @staticmethod
    def get_persona_by_name(context: Context, persona_name: str) -> Optional[Personality]:
        """
        根据名称获取指定的人格
        
        Args:
            context: Context对象
            persona_name: 人格名称
            
        Returns:
            指定名称的人格对象，如果不存在则返回None
        """
        try:
            personas = context.provider_manager.personas
            for persona in personas:
                if persona['name'] == persona_name:
                    return persona
            return None
        except Exception as e:
            logger.error(f"获取指定人格失败: {e}")
            return None