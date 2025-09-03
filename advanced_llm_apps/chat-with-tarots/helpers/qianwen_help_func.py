"""
塔罗牌项目通义千问帮助函数模块
用于处理塔罗牌相关的辅助功能
"""

import random
from typing import List, Dict, Any, Optional
import streamlit as st


class FallbackLLM:
    """
    回退LLM实现，当主要模型不可用时使用
    """
    def __init__(self, error_message: str):
        self.error_message = error_message
    
    def invoke(self, messages):
        """提供基础的回退响应"""
        return f"抱歉，AI模型暂时不可用：{self.error_message}。请稍后重试或检查配置。"


def generate_random_draw(num_cards: int, available_cards: List[str]) -> List[Dict[str, Any]]:
    """
    生成随机抽取的塔罗牌
    
    Args:
        num_cards: 要抽取的牌数
        available_cards: 可用的牌名列表
    
    Returns:
        包含抽取牌信息的字典列表
    """
    if num_cards > len(available_cards):
        raise ValueError(f"请求的牌数({num_cards})超过了可用牌数({len(available_cards)})")
    
    selected_cards = random.sample(available_cards, num_cards)
    drawn_cards = []
    
    for card_name in selected_cards:
        # 随机决定是否为逆位（30%概率）
        is_reversed = random.random() < 0.3
        
        card_info = {
            'name': card_name,
            'is_reversed': is_reversed
        }
        
        drawn_cards.append(card_info)
    
    return drawn_cards


def prepare_prompt_input(input_data: Dict[str, Any], card_meanings: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    准备用于LLM的提示输入
    
    Args:
        input_data: 包含cards和context的输入数据
        card_meanings: 卡牌含义字典
    
    Returns:
        格式化的提示输入字典
    """
    cards = input_data['cards']
    context = input_data['context']
    
    card_details_list = []
    symbolism_list = []
    
    for card_info in cards:
        card_name = card_info['name']
        is_reversed = card_info.get('is_reversed', False)
        
        # 获取卡牌含义
        if card_name in card_meanings:
            meaning_data = card_meanings[card_name]
            
            # 根据是否逆位选择含义
            if is_reversed:
                meaning = meaning_data.get('reversed', '逆位含义未知')
                position = '逆位'
            else:
                meaning = meaning_data.get('upright', '正位含义未知')
                position = '正位'
            
            symbolism = meaning_data.get('symbolism', '象征意义未知')
            
            card_detail = f"**{card_name}** ({position}): {meaning}"
            card_details_list.append(card_detail)
            symbolism_list.append(f"**{card_name}**: {symbolism}")
        else:
            # 如果找不到卡牌含义，提供默认信息
            position = '逆位' if is_reversed else '正位'
            card_detail = f"**{card_name}** ({position}): 含义待解读"
            card_details_list.append(card_detail)
            symbolism_list.append(f"**{card_name}**: 象征意义待解读")
    
    return {
        'card_details': '\n'.join(card_details_list),
        'context': context,
        'symbolism': '\n'.join(symbolism_list)
    }


def validate_card_data(df, required_columns: List[str]) -> bool:
    """
    验证卡牌数据的完整性
    
    Args:
        df: 卡牌数据DataFrame
        required_columns: 必需的列名列表
    
    Returns:
        验证是否通过
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"数据文件缺少必需的列: {', '.join(missing_columns)}")
        return False
    
    # 检查数据是否为空
    if df.empty:
        st.error("数据文件为空")
        return False
    
    return True


def format_card_image_path(card_name: str) -> str:
    """
    格式化卡牌图片路径
    
    Args:
        card_name: 卡牌名称
    
    Returns:
        格式化的图片文件名
    """
    # 移除特殊字符并转换为小写
    formatted_name = card_name.lower().replace(' ', '').replace('-', '')
    return f"{formatted_name}.jpg"


def display_error_message(error: Exception):
    """
    显示友好的错误信息
    
    Args:
        error: 异常对象
    """
    error_type = type(error).__name__
    
    if "ModuleNotFoundError" in str(error):
        st.error(f"🔧 模块导入错误: {error}")
        st.info("💡 请检查依赖是否正确安装：pip install -r qianwen_requirements.txt")
    elif "FileNotFoundError" in str(error):
        st.error(f"📁 文件未找到: {error}")
        st.info("💡 请确保所需的数据文件和图片文件存在于正确的路径")
    elif "API" in str(error) or "key" in str(error).lower():
        st.error(f"🔑 API配置错误: {error}")
        st.info("💡 请检查.env文件中的DASHSCOPE_API_KEY配置")
    else:
        st.error(f"❌ {error_type}: {error}")
        st.info("💡 请检查控制台输出获取更多错误信息")


# 常用的塔罗牌配置
TAROT_CONFIG = {
    'DEFAULT_CARDS': [3, 5, 7],
    'REVERSE_PROBABILITY': 0.3,
    'IMAGE_WIDTH': 150,
    'SUPPORTED_IMAGE_FORMATS': ['.jpg', '.jpeg', '.png']
}


def get_tarot_config(key: str, default=None):
    """
    获取塔罗牌配置值
    
    Args:
        key: 配置键名
        default: 默认值
    
    Returns:
        配置值
    """
    return TAROT_CONFIG.get(key, default)